import torch
import numpy as np

def get_batch(tokens, batch_size, context_length, device="mps"):
    
    max_start = tokens.shape[-1] - context_length

    starts = torch.randint(0, max_start,size=(batch_size,))

    inputs = np.stack([tokens[i: i + context_length] for i in starts])
    targets = np.stack([tokens[i + 1: i + context_length + 1] for i in starts])

    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)

    inputs = inputs.to(device).long()
    targets = targets.to(device).long()

    return (inputs, targets)
