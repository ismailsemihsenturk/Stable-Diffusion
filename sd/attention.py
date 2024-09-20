import torch
from torch import nn
from torch.nn import functional as F
import math

# d_embed is = channels because each pixel represented by many channels
class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3* d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed,d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x:torch.Tensor, causal_mask = False):
        # x: (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape

        batch_size,sequence_length, d_embed = input_shape

        intermim_shape = (batch_size,sequence_length,self.n_heads,self.d_head)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensors of shape (Batch_Size, Seq_Len, Dim)
        q,k,v = self.in_proj(x).chunk(3,dim=-1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch, Seq_Len, Head, Dim / H) -> (Batch_Size, Head, Seq_len, Dim / H)
        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)

        # (Batch_Size, Head, Seq_Len, Dim / H) @ (Batch_Size, Head, Dim / H, Seq_Len) -> (Batch_Size, Head, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf) 
        
        # Divide by d_k (Dim / H). 
        # (Batch_Size, Head, Seq_Len, Seq_Len) -> (Batch_Size, Head, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head) 

        # (Batch_Size, Head, Seq_Len, Seq_Len) -> (Batch_Size, Head, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1) 

        # (Batch_Size, Head, Seq_Len, Seq_Len) @ (Batch_Size, Head, Seq_Len, Dim / H) -> (Batch_Size, Head, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, Head, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, Head, Dim / H)
        output = output.transpose(1, 2) 

        # (Batch_Size, Seq_Len, Head, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output) 
        
        # (Batch_Size, Seq_Len, Dim)
        return output
        

# 1. Understanding the Two Attention Mechanisms
    # They both derived from SelfAttention.
    # 1.1. Cross Attention
    # Definition: Cross Attention is a mechanism where queries (Q) from one source attend to keys (K) and values (V) from another source. This is distinct from self-attention, where Q, K, and V all come from the same source.

# Use Cases:
    # Encoder-Decoder Architectures: In models like the original Transformer for machine translation, the decoder uses cross attention to focus on the encoder's output.
    # Stable Diffusion: Utilizes cross attention to integrate information from different modalities or layers, ensuring coherent generation based on varied inputs.

# Functionality:
    # Information Flow: Facilitates the flow of information between different parts of the model, allowing one component to influence another.
    # Contextual Integration: Helps in integrating context from different sources, enhancing the model's ability to generate relevant and coherent outputs.

# 1.2. Group Query Attention
    # Definition: Group Query Attention is a specialized form of attention where queries are divided into distinct groups. Each group focuses on different aspects or subsets of the data, allowing the model to capture diverse features more effectively.

# Use Cases:
    # Vision-Language Models (VLMs): Enhances the model's ability to handle multi-modal data by segregating queries into groups that can independently attend to different modalities.
    # Object Detection Models (e.g., DETR): Uses group queries to detect multiple objects by assigning different query groups to different object categories or regions.

# Functionality:
    # Diversity in Attention: By grouping queries, the model can attend to various features or regions independently, promoting feature diversity.
    # Computational Efficiency: Allows parallel processing of query groups, potentially reducing computational overhead and improving efficiency.

# While Group Query Attention and Cross Attention are both powerful mechanisms within the realm of attention-based models, they serve distinct purposes tailored to the specific needs of different architectures and tasks:
    # Cross Attention is essential for enabling interaction and information flow between distinct data sources within encoder-decoder frameworks. It ensures that the decoder can effectively utilize the encoded information from the encoder, which is crucial for tasks like translation, summarization, and image generation conditioned on textual prompts.
    # Group Query Attention excels in scenarios where diverse feature extraction within a single data source is required, such as in object detection or multi-modal data handling. It enhances the model's ability to process complex, multifaceted data efficiently by segregating queries into groups that can independently focus on different aspects.

# Recommendation:
    # Maintain Cross Attention in encoder-decoder architectures where inter-source interaction is pivotal.
    # Utilize Group Query Attention in components of the model that benefit from feature diversity and efficient processing within a single data source, such as in the encoder or within specific modules handling multi-modal data.
class CrossAttention(nn.Module):
    
    # d_embed = queries
    # d_cross = key and values from different source
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    

    def forward(self,x,y):
        # x (latent): (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)
        # 77 Seq_Len of our prompt is 77
        # 768 because each embedding is of size 768 

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2) 
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()
        
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output