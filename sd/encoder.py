import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


# It's sequence of models that reduce the dimension but increase the features of the latent space.
# With Conv2D layers we are reducing the size of the image 
# With Residual blocks we are increasing the features of the images.
class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            # We have same height and width dimensions because of the padding
            nn.Conv2d(3,128, kernel_size=3, padding=1),

            # Combination of convolutions and normalizations
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128,128),
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128,128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2)
            nn.Conv2d(128,128,kernel_size=3, stride=2,padding=0),

            # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(128,256),
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256,256),

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
            nn.Conv2d(256,256,kernel_size=3, stride=2,padding=0),

            # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(256,512),
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512,512),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32,512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.SiLU(), 


            # Because the padding=1, it means the width and height will increase by 2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(512,8,kernel_size=3,padding=1),

            # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(8,8, kernel_size=1,padding=0),
        )

    def forward(self,x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: Has the same size with the output of the encoder = (Batch_Size, Out_Channels, Height / 8, Width / 8)

        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                # This code says can you add a layer of pixels to the right and bottom side of the image only. Because when we do stride whe set the padding zero in init func but we want a asymmetrical padding
                # Padding at downsampling should be asymmetric
                # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                # Pad with zeros on the right and bottom.
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0,1,0,1))
            x = module(x)

        # (Batch_Size, 8, Height, Height / 8, Width / 8) -> Two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x,2,dim=1)

        # (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30,20)
        # (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()

        # (Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()

        # Converting Normal to Standart Normal
        # Z = (X - mean) / std 
        # X = mean + std * Z
        # Z=N(0,1) -> X=N(mean, variance)?
        # X = mean + stdev * Z -> Transsforms N(0,1) Distribution to N(mean,variance)
        x = mean + stdev * noise

        # Scale the output by a constant
         # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        return x
