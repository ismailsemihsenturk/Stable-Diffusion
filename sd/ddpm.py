import torch
import numpy as np

class DDPMSampler:

    def __init__(
            self,
            generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120
        ):
        # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        # num_training_steps is how many pieces that we want to divide this linspace (linear space) because it's a gaussian maskov chain.
        # We are going to use it for Forward Process (noisify the image)
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        # We are going to use alphas to go from where we are to any timestep to get a noisy image. It's also used for Forward Process
        self.alphas = 1.0 - self.betas
        # [alpha_0, alppha_0 * alpha_1, alpha_0 * alpha_1 * alpha_2...] cumulative multiplication
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator

        self.num_train_timesteps = num_training_steps
        # Training steps is 0 to 999 but we are going to use it reverse 999 to 0 for the Reverse Process
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())


    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t


    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        return variance



    def set_strength(self, strength=1):
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
        


    # Equations from https://arxiv.org/pdf/2006.11239.pdf
    # ----------------------------------
    # FOR TRAINING WE DID (ALGORITHM 1 IN THE PAPER):
    # || E - E_θ(((a_t)^1/2)* x_0 + ((1-a_t)^1/2) * E, t ||^2
    # a for alpha 
    # e for epsilon
    # ----------------------------------
    # ----------------------------------
    # FOR SAMPLING (FIRST FORMULA IN THE PAPER):
    # p_θ(X_(t-1) | X_t) = N(X_(t-1); μ_θ(X_t, t), Σ_θ(X_t, t)) 
    # This formula says how much noise do you have in given timestep. It's not saying how to predict the MEAN and VARIANCE. 
    # So our goal is sample x_t-1 ~ p_θ(X_(t-1) | X_t). WE CAN DO THIS WITH 2 WAYS:
    # FIRST METHOD:
    # WE ARE GOING TO USE 6TH FORMULA FOR SAMPLING:
        # First: We are going to define our coefficients. Alpha, Alpha bar, Beta and Beta bar
            # α_t:= 1 − β_t
            # β˜_t:= (1 − α¯t−1 / 1 − α¯t)* β
            # β_t :=  σ^2_t
        # Second (FORMULA 15): we are going to predict x_0 from our coefficients. 
        # Third (FORMULA 7): We are going to compute the coefficients with the x_0 we computed. Thus we are going to find the MEAN and the VARIANCE.
    #----------------------------------    
    # SECOND METHOD:
    # WE ARE GOING TO USE 11TH FORMULA FOR SAMPLING:  
    # x_t-1 = ((a_t)^-1/2) * (x_t - (B_t / ((1-a_t)^1/2)) * E_θ(x_t,t)) + sigma_t * z where z ~ N(0,1)
    # This theta (θ) comes from the training formula where we do gradient descent
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 3. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 4. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # 5. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            variance = (self._get_variance(t) ** 0.5) * noise
        
        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # the variable "variance" is already multiplied by the noise N(0, 1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
        





    # q(x_t | x_0) = N(x_t; ((a_t)^1/2)x_0, (1-a_t)I)
    # MEAN: ((a_t)^1/2)x_0
    # Variance: (1-a_t) 
    def add_noise(self, original_samples: torch.FloatTensor, timestep: torch.IntTensor) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # we want to add this alpha to latents so we need some dimensions. That's why we used flatten() func. And unsqueeze() until we have the same dimensions.
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sample from q(x_t | x_0) as in equation (4) of https://arxiv.org/pdf/2006.11239.pdf
        # Because N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # here mu = sqrt_alpha_prod * original_samples and sigma = sqrt_one_minus_alpha_prod
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
