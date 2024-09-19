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
        