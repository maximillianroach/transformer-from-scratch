import torch
import math

def learning_rate_schedule(t, alpha_max, alpha_min, Tw, Tc):
    if t < Tw:
        lr = (t / Tw) * alpha_max
    elif t >= Tw and t <= Tc:
        lr = alpha_min + (0.5) * (1 + math.cos(math.pi * (t - Tw) / (Tc - Tw))) * (alpha_max - alpha_min)
    else:
        lr = alpha_min
    return lr