import torch
from einops import rearrange, einsum, reduce
import torch.nn as nn
from .scaled_dot_product_attention import scaled_dot_product_attention
from .rope import RotaryPositionalEmbedding

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_h = d_model // num_heads
        self.num_heads = num_heads

        self.wO = nn.Parameter(torch.empty(size=(d_model, d_model)))
        nn.init.normal_(self.wO, mean=0, std=1)

        self.wQ = nn.Parameter(torch.empty(size=(d_model, d_model)))
        nn.init.normal_(self.wQ, mean=0, std=1)

        self.wK = nn.Parameter(torch.empty(size=(d_model, d_model)))
        nn.init.normal_(self.wK, mean=0, std=1)

        self.wV = nn.Parameter(torch.empty(size=(d_model, d_model)))
        nn.init.normal_(self.wV, mean=0, std=1)

    def forward(self, X):
        Q = einsum(X, self.wQ, '... n d_in, d_out d_in -> ... n d_out')
        K = einsum(X, self.wK, '... n d_in, d_out d_in -> ... n d_out')
        V = einsum(X, self.wV, '... n d_in, d_out d_in -> ... n d_out')

        seq_len = X.shape[-2]
        # we rearrange so matrix multiplication is performed per head
        Q = rearrange(Q, '... n (h d_h) -> ... h n d_h', h=self.num_heads)
        K = rearrange(K, '... n (h d_h) -> ... h n d_h', h=self.num_heads)
        V = rearrange(V, '... n (h d_h) -> ... h n d_h', h=self.num_heads)


        # qkT will be seq_len by seq_len
        ones = torch.ones(size=(seq_len, seq_len), dtype=torch.bool)

        mask = torch.tril(ones)

        out = scaled_dot_product_attention(K, Q, V, mask=mask)

        out = rearrange(out, '... h n d_h -> ... n (h d_h)')

        res = einsum(self.wO, out, 'd_out d_in, ... n d_in -> ... n d_out')

        return res

class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_h = d_model // num_heads
            self.num_heads = num_heads
            self.d_model = d_model
    
            self.wO = nn.Parameter(torch.empty(size=(d_model, d_model)))
            nn.init.normal_(self.wO, mean=0, std=1)
    
            self.wQ = nn.Parameter(torch.empty(size=(d_model, d_model)))
            nn.init.normal_(self.wQ, mean=0, std=1)
    
            self.wK = nn.Parameter(torch.empty(size=(d_model, d_model)))
            nn.init.normal_(self.wK, mean=0, std=1)
    
            self.wV = nn.Parameter(torch.empty(size=(d_model, d_model)))
            nn.init.normal_(self.wV, mean=0, std=1)

    def forward(self, X, theta, max_seq_len, token_positions):
        rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_model // self.num_heads, max_seq_len=max_seq_len)

        Q = einsum(X, self.wQ, '... n d_in, d_out d_in -> ... n d_out')
        K = einsum(X, self.wK, '... n d_in, d_out d_in -> ... n d_out')
        V = einsum(X, self.wV, '... n d_in, d_out d_in -> ... n d_out')

        seq_len = X.shape[-2]
        # we rearrange so matrix multiplication is performed per head
        Q = rearrange(Q, '... n (h d_h) -> ... h n d_h', h=self.num_heads)
        K = rearrange(K, '... n (h d_h) -> ... h n d_h', h=self.num_heads)
        V = rearrange(V, '... n (h d_h) -> ... h n d_h', h=self.num_heads)

        K = rope(K, token_positions)
        Q = rope(Q, token_positions)


        # qkT will be seq_len by seq_len
        ones = torch.ones(size=(seq_len, seq_len), dtype=torch.bool)

        mask = torch.tril(ones)

        out = scaled_dot_product_attention(K, Q, V, mask=mask)

        out = rearrange(out, '... h n d_h -> ... n (h d_h)')

        res = einsum(self.wO, out, 'd_out d_in, ... n d_in -> ... n d_out')

        return res

    