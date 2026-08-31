import torch
import torch.nn as nn
from .linear import Linear
from einops import rearrange, einsum, reduce

class SwiGLU(nn.Module):
    def __init__(self, d_ff, d_model, device=None, dtype=None):
        super().__init__()

        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=None, dtype=None))
        nn.init.normal_(self.w1, mean=0, std=1)

        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=None, dtype=None))
        nn.init.normal_(self.w3, mean=0, std=1)

        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=None, dtype=None))
        nn.init.normal_(self.w2, mean=0, std=1)

    def forward(self, x):
        w1_product = einsum(x, self.w1, '... d_model, d_ff d_model -> ... d_ff')

        silu = w1_product * torch.sigmoid(w1_product)

        w3_product = einsum(x, self.w3, '... d_model, d_ff d_model -> ... d_ff')

        compwise_product = silu * w3_product

        res = einsum(compwise_product, self.w2, '... d_ff, d_model d_ff -> ... d_model')

        return res

class SiLU(nn.Module):
    def __init__(self, d_ff, d_model, device=None, dtype=None):
        super().__init__()
        
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=None, dtype=None))
        nn.init.normal_(self.w1, mean=0, std=1)

        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=None, dtype=None))
        nn.init.normal_(self.w2, mean=0, std=1)

    def forward(self, x):
        w1_product = einsum(x, self.w1, '... d_model, d_ff d_model -> ... d_ff')
        silu = w1_product * torch.sigmoid(w1_product)
        res = einsum(silu, self.w2, '... d_ff, d_model d_ff -> ... d_model')
        return res