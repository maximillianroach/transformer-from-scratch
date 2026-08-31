import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        thetas = torch.ones(size=(max_seq_len, d_k // 2)) * torch.arange(max_seq_len).unsqueeze(1)
        # get the powers in the denominator
        powers = (2 * torch.arange(1, d_k // 2 + 1) - 2) / d_k
        # raise the constant theta to the powers
        thetas = thetas / theta ** powers

        sines = torch.sin(thetas)
        cosines = torch.cos(thetas)

        self.register_buffer("sines", sines, persistent=False)
        self.register_buffer("cosines", cosines, persistent=False)

        self.d_k = d_k

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cosine = self.cosines[token_positions]
        sine = self.sines[token_positions]

        x_pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)

        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]

        

        res_0 = cosine * x0 - sine * x1
        res_1 = sine * x0 + cosine * x1

        res = torch.stack((res_0, res_1), dim=-1).reshape(*x.shape)
        return res


        
