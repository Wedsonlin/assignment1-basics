import torch
from torch import nn
from math import sqrt
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        sigma = sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.W, 0, sigma, -3*sigma, 3*sigma)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        nn.init.trunc_normal_(self.embedding_matrix, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
            X=(x_1,x_2,...): LongTensor, W: Tensor
            W[X] -> 用X中的每个元素x_i去索引W,得(W[x_1],W[x_2],...)
        '''
        return self.embedding_matrix[token_ids]
            
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS_x = torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
        x = x * RMS_x * self.g
        return x.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int=None, device=None, dtype=None):
        super().__init__()
        if not d_ff:
            d_ff = round(8/3 * d_model / 64) * 64
        self.W1 = nn.Parameter(torch.empty(d_ff,d_model,device=device,dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model,d_ff,device=device,dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(d_ff,d_model,device=device,dtype=dtype))

        sigma = sqrt(2/(d_ff+d_model))
        nn.init.trunc_normal_(self.W1, 0, sigma, -3*sigma, 3*sigma)
        nn.init.trunc_normal_(self.W2, 0, sigma, -3*sigma, 3*sigma)
        nn.init.trunc_normal_(self.W3, 0, sigma, -3*sigma, 3*sigma)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        W1x = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        W3x = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        SiLU_W1x = W1x*torch.sigmoid(W1x)
        return einsum(self.W2, SiLU_W1x*W3x, "d_model d_ff, ... d_ff -> ... d_model")
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        b = torch.ones(d_k,device=device) * theta
        e = torch.Tensor(d_k,device=device)
        k = 1
        for i in range(d_k,2):
            e[i] = -(2*k-2)/d_k
            e[i+1] = -(2*k-2)/d_k
            k += 1

        self.d_k = d_k
        self.s = torch.empty(max_seq_len,d_k,device=device)
        for i in range(max_seq_len):
            self.s[i] = torch.sin(torch.pow(b,(i+1)*e))
        
        self.c = torch.Tensor(max_seq_len,d_k,device=device)
        for i in range(max_seq_len):
            self.c[i] = torch.sin(torch.pow(b,(i+1)*e))

        self.register_buffer(name="RoPE buffer", tensor=None, persistent=False)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x2 = torch.Tensor(x)

        sign = torch.tensor([-1, 1], device=x2.device, dtype=x2.dtype)
        x2.reshape(*x2.shape[:-1], self.d_k // 2, 2).flip(-1) * sign
        x2 = x2.reshape(*x2.shape[:-1], self.d_k)

        return x * self.s[token_positions] + x2 * self.c[token_positions]
