import torch
import math

def gradient_clipping(params, max_l2_norm):
    combined_l2_norm = 0
    eps = 1e-6

    for p in params:
        if p.grad is None:
            continue

        combined_l2_norm += torch.sum(p.grad**2)

    combined_l2_norm = math.sqrt(combined_l2_norm)

    if combined_l2_norm > max_l2_norm:
        for p in params:
            if p.grad is None:
                continue

            p.grad = p.grad * ((max_l2_norm) / (combined_l2_norm + eps))