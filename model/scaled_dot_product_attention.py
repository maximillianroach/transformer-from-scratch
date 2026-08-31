import torch
from einops import reduce, rearrange, einsum
from .softmax import softmax
import math

def scaled_dot_product_attention(keys, queries, values, mask=None):
    d_k = keys.shape[-1]

    qkT = einsum(queries, keys, '... n d_k, ... m d_k -> ... n m')

    soft_inner = qkT / math.sqrt(d_k)

    if mask is not None:
            soft_inner = soft_inner.masked_fill(~mask, float("-inf"))

    soft = softmax(soft_inner, dim=-1)
    res = einsum(soft, values, '... n m, ... m d_v -> ... n d_v')

    return res




