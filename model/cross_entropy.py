from .softmax import softmax
import torch

def cross_entropy(logits, targets):
    batch_size = logits.shape[0]

    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)

    n = logits.shape[0]
    target_logits = logits[torch.arange(n), targets]
    m = torch.amax(logits, dim=-1, keepdim=True)
    shifted = logits - m

    s = torch.sum(torch.exp(shifted), dim=-1)
    l = torch.log(s)
    ce = m.squeeze(-1) + l - target_logits

    res = torch.mean(ce)

    return res

