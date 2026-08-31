import torch
import typing
import os

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": iteration
    }

    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer): 

    checkpoint = torch.load(src)

    model_state = checkpoint["model_state"]
    optimizer_state = checkpoint["optimizer_state"]
    step = checkpoint["step"]

    model_state = {
            k.replace("_orig_mod.", "", 1): v
            for k, v in model_state.items()
    }

    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

    return step

    
