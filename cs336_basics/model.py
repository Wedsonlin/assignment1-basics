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

def SiLU(x:torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
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
        return einsum(self.W2, SiLU(W1x)*W3x, "d_model d_ff, ... d_ff -> ... d_model")
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k

        k = torch.arange(0, d_k//2, 1, dtype=torch.float32, device=device)
        freq = torch.pow(theta, -2 * k / d_k)
        position = torch.arange(0, max_seq_len, 1, dtype=torch.float32, device=device)
        angle = torch.outer(position, freq)

        # Register sin/cos as buffers for proper device management
        self.register_buffer("s", torch.sin(angle).repeat_interleave(2, dim=-1))
        self.register_buffer("c", torch.cos(angle).repeat_interleave(2, dim=-1))

        # Register sign pattern as buffer
        self.register_buffer("sign", torch.tensor([-1, 1], dtype=torch.float32))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        Applies rotary positional embeddings to input tensor.
        Transforms (1,2,3,4) -> (-2,1,-4,3) pattern for rotation.
        '''
        x2 = x.clone()

        x2 = x2.reshape(x.shape[:-1] + (self.d_k//2, 2))
        x2 = x2.flip(dims=[-1])
        x2 = x2 * self.sign
        x2 = x2.reshape(x.shape[:-1] + (self.d_k,))

        return x * self.c[token_positions] + x2 * self.s[token_positions]

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    max_element = x.amax(dim=i,keepdim=True) # avoid overflow
    exp_y = torch.exp(x-max_element)
    return exp_y / exp_y.sum(dim=i,keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor=None):
    '''
    Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) @ V

    Computes attention weights based on query-key similarity and applies them to values.
    For each query q_i, the output o_i is a weighted sum of values: sum_j w_{ij} * v_j

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor (True = attend, False = mask out)
    Return:
        Float[Tensor, " ... queries d_v"]: Attention output
    '''
    d_k = Q.shape[-1]

    K_T = rearrange(K, "... keys d_k -> ... d_k keys")
    QK_T = einsum(Q, K_T, "... queries d_k, ... d_k keys -> ... queries keys") / sqrt(d_k)
    if mask is not None:
        QK_T = QK_T.masked_fill(~mask, -torch.inf)
    weight = softmax(QK_T, -1)
    return einsum(weight, V, "... queries keys, ... keys d_v -> ... queries d_v")

class MultiheadSelfAttention(nn.Module):
    '''
    Causal multi-head self-attention
    '''
    def __init__(self, d_model: int, num_heads: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta

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

        # Cache for RoPE and causal mask
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)

        if max_seq_len:
            causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        else:
            causal_mask = None

        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        Q = einsum(self.W_Q, x, "hd_k d_model, ... S d_model -> ... S hd_k")
        K = einsum(self.W_K, x, "hd_k d_model, ... S d_model -> ... S hd_k")
        V = einsum(self.W_V, x, "hd_v d_model, ... S d_model -> ... S hd_v")

        # Rearrange heads as part of batch dimension
        Q = rearrange(Q, "... S (h d_k) -> ... h S d_k", h=self.num_heads, d_k=self.d_k)
        K = rearrange(K, "... S (h d_k) -> ... h S d_k", h=self.num_heads, d_k=self.d_k)
        V = rearrange(V, "... S (h d_v) -> ... h S d_v", h=self.num_heads, d_v=self.d_v)

        # Apply RoPE if requested (cache to avoid recreation)
        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        seq_len = x.shape[-2]
        if self.causal_mask is None or self.causal_mask.shape[-1] < seq_len :
            self.causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        mask = self.causal_mask[:seq_len, :seq_len]

        heads = scaled_dot_product_attention(Q, K, V, mask)
        head = rearrange(heads, "... h S d_v -> ... S (h d_v)", h=self.num_heads, d_v=self.d_v)

        return einsum(self.W_O, head, "d_model hd_v, ... S hd_v -> ... S d_model")

class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int=None,
                 theta: float=None,
                 weights: dict[str, torch.Tensor]=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.rmsnorm1 = RMSNorm(d_model)
        self.multihead_self_attention = MultiheadSelfAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta)
        self.rmsnorm2 = RMSNorm(d_model)
        self.positionwise_feedforward = SwiGLU(d_model, d_ff)

        # Cache for token positions
        position_ids = torch.arange(max_seq_len)
        self.register_buffer("position_ids", position_ids, persistent=False)

        if weights:
            self.multihead_self_attention.load_state_dict({
                "W_Q": weights['attn.q_proj.weight'],
                "W_K": weights['attn.k_proj.weight'],
                "W_V": weights['attn.v_proj.weight'],
                "W_O": weights['attn.output_proj.weight']
            })
            self.rmsnorm1.load_state_dict({
                'g': weights['ln1.weight']
            })
            self.positionwise_feedforward.load_state_dict({
                "W1": weights['ffn.w1.weight'],
                "W2": weights['ffn.w2.weight'],
                "W3": weights['ffn.w3.weight'],
            })
            self.rmsnorm2.load_state_dict({
                'g': weights['ln2.weight']
            })

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[-2]
        token_position = self.position_ids[:seq_len]

        # Sublayer 1: Causal multihead self-attention with RoPE
        x = x + self.multihead_self_attention(self.rmsnorm1(x), token_position)
        # Sublayer 2: Position-wise feed-forward
        x = x + self.positionwise_feedforward(self.rmsnorm2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 weights: dict[str, torch.Tensor]=None):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights

        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # Create transformer blocks
        if self.weights:
            transformer_list = [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    max_seq_len=self.context_length,
                    theta=self.rope_theta,
                    weights={
                        "attn.q_proj.weight": self.weights[f"layers.{i}.attn.q_proj.weight"],
                        "attn.k_proj.weight": self.weights[f"layers.{i}.attn.k_proj.weight"],
                        "attn.v_proj.weight": self.weights[f"layers.{i}.attn.v_proj.weight"],
                        "attn.output_proj.weight": self.weights[f"layers.{i}.attn.output_proj.weight"],
                        "ln1.weight": self.weights[f"layers.{i}.ln1.weight"],
                        "ffn.w1.weight": self.weights[f"layers.{i}.ffn.w1.weight"],
                        "ffn.w2.weight": self.weights[f"layers.{i}.ffn.w2.weight"],
                        "ffn.w3.weight": self.weights[f"layers.{i}.ffn.w3.weight"],
                        "ln2.weight": self.weights[f"layers.{i}.ln2.weight"]
                    }
                )
                for i in range(self.num_layers)
            ]
        else:
            transformer_list = [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    max_seq_len=self.context_length,
                    theta=self.rope_theta
                )
                for _ in range(self.num_layers)
            ]

        self.transformer_blocks = nn.Sequential(*transformer_list)

        self.rmsnorm = RMSNorm(d_model=d_model)
        
        self.output_embedding = Linear(in_features=d_model,out_features=vocab_size)

        if self.weights:
            self.token_embedding.load_state_dict({
                "embedding_matrix": self.weights['token_embeddings.weight']
            })
            self.rmsnorm.load_state_dict({
                "g": self.weights['ln_final.weight']
            })
            self.output_embedding.load_state_dict({
                "W": self.weights['lm_head.weight']
            })

    def forward(self, x:torch.Tensor):
        x = self.token_embedding(x)
        x = self.transformer_blocks(x)
        x = self.rmsnorm(x)
        x = self.output_embedding(x)
        return x # PyTorch models output logits; loss functions apply softmax internally for numerical stability

        