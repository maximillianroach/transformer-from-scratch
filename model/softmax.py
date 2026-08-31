import torch
import torch.nn as nn
from einops import reduce, einsum, rearrange

def softmax(v: torch.Tensor, dim: int):
    m = torch.amax(v, dim=dim, keepdim=True)

    shifted = v - m

    exponents = torch.exp(shifted)

    s = torch.sum(exponents, dim=dim, keepdim=True)

    res = exponents / s

    return res