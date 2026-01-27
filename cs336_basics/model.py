import torch
from torch import nn
from math import sqrt
from einops import einsum, rearrange

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

        k = torch.arange(0,d_k//2,1,dtype=torch.float32, device=device) # (0,1,...,d_k//2)
        freq = torch.pow(theta,-2 * k / d_k)
        position = torch.arange(0,max_seq_len,1,dtype=torch.float32) # position从0开始....
        angle = torch.outer(position,freq) # a,b是列向量,outer(a,b)=<a,b^T>

        self.s = torch.sin(angle)
        self.c = torch.cos(angle)

        self.s = self.s.repeat_interleave(2,dim=-1) # repeat elements in given dimension
        self.c = self.c.repeat_interleave(2,dim=-1)
        self.d_k = d_k

        self.register_buffer(name="RoPE buffer", tensor=None, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
            torch.Tensor(x): create a reference of x, use same storage
            torch.tensor(x): copy data from x, use different storage, discard autograd relation
            y=x.clone(): copy data from x, use different storage, reserve autograd relation dy/dx=1
        '''
        x2 = x.detach().clone()

        '''
            (1,2,3,4) -> (-2,1,-4,3)
        '''
        sign = torch.tensor([-1, 1], device=x2.device, dtype=x2.dtype)
        x2 = x2.reshape(x.shape[:-1] + (self.d_k//2,2))
        x2 = x2.flip(dims=[-1])
        x2 = x2 * sign
        x2 = x2.reshape(x.shape[:-1] + (self.d_k,))

        return x * self.c[token_positions] + x2 * self.s[token_positions]

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    max_element = x.amax(dim=i,keepdim=True) # avoid overflow
    exp_y = torch.exp(x-max_element)
    return exp_y / exp_y.sum(dim=i,keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor=None):
    '''
    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Return:
        softmax(QK^T / sqrt(d_k)) @ V
        根据 Q 与 K 的关联性(内积大小)得到 values 关于 q_i 的权重,
        那么 q_i 对应的输出 o_i 为 values 的加权和, 即 sum_{i=j}^{values} w_{ij}*v_j
    '''
    d_k = Q.shape[-1]

    K_T = rearrange(K, "... keys d_k -> ... d_k keys")
    QK_T = einsum(Q,K_T,"... queries d_k, ... d_k keys -> ... queries keys") / sqrt(d_k)
    if mask is not None:
        QK_T = QK_T.masked_fill(~mask,-torch.inf) # 当mask[i,j] == True时,i-th query和j-th key有关联 
    weight = softmax(QK_T,-1)
    return einsum(weight,V,"... queries keys, ... keys d_v -> ... queries d_v")

class MultiheadSelfAttention(nn.Module):
    '''
        reduced casual multi-head self attention
    '''
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.num_heads = num_heads

        self.W_Q = nn.Parameter(torch.empty(num_heads*self.d_k, d_model, device=device, dtype=dtype))
        self.W_K = nn.Parameter(torch.empty(num_heads*self.d_k, d_model, device=device, dtype=dtype))
        self.W_V = nn.Parameter(torch.empty(num_heads*self.d_v, d_model, device=device, dtype=dtype))
        self.W_O = nn.Parameter(torch.empty(d_model, num_heads*self.d_v, device=device, dtype=dtype))

        sigma = sqrt(2/(num_heads*self.d_k+d_model))
        nn.init.trunc_normal_(self.W_Q, 0, sigma, -3*sigma, 3*sigma)
        nn.init.trunc_normal_(self.W_K, 0, sigma, -3*sigma, 3*sigma)

        sigma = sqrt(2/(num_heads*self.d_v+d_model))
        nn.init.trunc_normal_(self.W_V, 0, sigma, -3*sigma, 3*sigma)
        nn.init.trunc_normal_(self.W_O, 0, sigma, -3*sigma, 3*sigma)
    
    def forward(self, x: torch.Tensor, max_seq_len: int=None, theta: float=None, token_positions: torch.Tensor=None) -> torch.Tensor:
        Q = einsum(self.W_Q, x, "hd_k d_model, ... S d_model -> ... S hd_k")
        K = einsum(self.W_K, x, "hd_k d_model, ... S d_model -> ... S hd_k")
        V = einsum(self.W_V, x, "hd_v d_model, ... S d_model -> ... S hd_v")

        # 把 head 作为 batch dimension 的一部分
        Q = rearrange(Q, "... S (h d_k) -> ... h S d_k", h=self.num_heads, d_k=self.d_k)
        K = rearrange(K, "... S (h d_k) -> ... h S d_k", h=self.num_heads, d_k=self.d_k)
        V = rearrange(V, "... S (h d_v) -> ... h S d_v", h=self.num_heads, d_v=self.d_v)

        if token_positions != None:
            RoPE = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
            Q = RoPE(Q, token_positions)
            K = RoPE(K, token_positions)

        # mask:(queries keys) = (S S) 自动广播 (... S S)
        mask = torch.tril(torch.ones(x.shape[-2],x.shape[-2],dtype=torch.bool,device=x.device))

        heads = scaled_dot_product_attention(Q,K,V,mask)
        head = rearrange(heads, "... h S d_v -> ... S (h d_v)", h=self.num_heads, d_v=self.d_v)

        return einsum(self.W_O, head, "d_model hd_v, ... S hd_v -> ... S d_model")
    

def transformer_block(x: torch.Tensor, d_model: int, num_heads: int, d_ff: int, max_seq_len: int=None, theta: float=None , weights: dict[str, torch.Tensor]=None):
    rmsnorm1 = RMSNorm(d_model)
    multihead_self_attention = MultiheadSelfAttention(d_model,num_heads)
    rmsnorm2 = RMSNorm(d_model)
    positionwise_feedforward = SwiGLU(d_model, d_ff)
    
    if weights:
        multihead_self_attention.load_state_dict({
            "W_Q":weights['attn.q_proj.weight'],
            "W_K":weights['attn.k_proj.weight'],
            "W_V":weights['attn.v_proj.weight'],
            "W_O":weights['attn.output_proj.weight']
        })
        rmsnorm1.load_state_dict({
            'g':weights['ln1.weight']
        })
        positionwise_feedforward.load_state_dict({
            "W1":weights['ffn.w1.weight'],
            "W2":weights['ffn.w2.weight'],
            "W3":weights['ffn.w3.weight'],
        })
        rmsnorm2.load_state_dict({
            'g':weights['ln2.weight']
        })

    token_position = torch.arange(0,x.shape[-2],1,device=x.device)
    y = x + multihead_self_attention(rmsnorm1(x), max_seq_len, theta, token_position)
    y = y + positionwise_feedforward(rmsnorm2(y))

    return y

class TransformerLM(nn.modules):
    def __init__(self, vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, torch.Tensor]=None):
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights

        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.rmsnorm = RMSNorm(d_model=d_model)
        self.linear = Linear(in_features=d_model, out_features=d_model)

    def forward(self, x:torch.Tensor):
        x = self.token_embedding(x)
        for _ in range(self.num_layers):
            x = transformer_block(x,self.d_model, )
    
        