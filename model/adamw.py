import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-5, weight_decay=1e-2):
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lam = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # compute gradient of the loss with respect to parameter p
                grad = p.grad.data

                state = self.state[p]
                t = state.get("t", 0) + 1

                # learning rate decay
                alpha = lr * ((math.sqrt(1 - math.pow(beta2, t))) / (1 - math.pow(beta1, t)))

                # apply weight decay
                p.data = p.data - lr * lam * p.data

                # update first moment estimate
                m = state.get("m", torch.zeros_like(p))
                m = beta1 * m + (1 - beta1) * grad

                # update second moment estimate
                v = state.get("v", torch.zeros_like(p))
                v = beta2 * v + (1 - beta2) * grad ** 2

                # apply moment-adjusted weight updates
                p.data = p.data - alpha * (m / (torch.sqrt(v) + eps))

                state["t"] = t
                state["m"] = m
                state["v"] = v

        return loss






