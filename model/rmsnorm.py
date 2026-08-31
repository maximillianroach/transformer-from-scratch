import torch
import torch.nn as nn
from einops import rearrange, reduce, einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()

        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # done to prevent overflow when squaring
        x = x.to(torch.float32)

        squares = x ** 2
        means = reduce(squares, '... din -> ... ()', 'mean')

        rms = torch.sqrt(means + self.eps)
        rms = x / rms

        rms = rms * self.g

        return rms.to(in_dtype)
    


