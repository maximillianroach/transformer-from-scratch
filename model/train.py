import argparse
import numpy as np
import torch
from pathlib import Path
from .tokenizer import Tokenizer
from .transformer import Transformer
from .adamw import AdamW
from .get_batch import get_batch
from .cross_entropy import cross_entropy
from .gradient_clipping import gradient_clipping
from .learning_rate_schedule import learning_rate_schedule
from .checkpointing import save_checkpoint
from .decoding import decode
import wandb


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embed-dim",
        type=int,
    )

    parser.add_argument(
        "--lr",
        type=float
    )

    parser.add_argument(
        "--num-layers",
        type=int
    )

    parser.add_argument(
        "--num-heads",
        type=int
    )

    parser.add_argument(
        "--weight-decay",
        type=float
    )

    parser.add_argument(
        "--d-model",
        type=int
    )

    parser.add_argument(
        "--d-ff",
        type=int
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True
    )

    parser.add_argument(
        "--num-steps",
        type=int
    )

    parser.add_argument(
        "--context-length",
        type=int
    )

    parser.add_argument(
        "--batch-size",
        type=int
    )

    parser.add_argument(
        "--min-lr",
        type=float
    )

    parser.add_argument(
        '--warmup-steps',
        type=int
    )

    parser.add_argument(
        '--checkpoint-path',
        type=str
    )

    return parser.parse_args()

# take the vocab and merges files from before and use them to tokenize train and validation datasets
def tokenize(vocab_path, merges_path, data_path, out_path):
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

    with open(data_path, "r") as f:
        text = f.read()

    tokens = tokenizer.encode(text)

    np.array(tokens, dtype=np.uint16).tofile(out_path)

def train():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_files("data/tinystories_vocab.pkl", "data/tinystories_merges.pkl")
    vocab_size = len(tokenizer.vocab)

    tokens = np.memmap("data/tinystories_tokens_train.bin", dtype=np.uint16, mode="r")
    val_tokens = np.memmap("data/tinystories_tokens_val.bin", dtype=np.uint16, mode="r")

    model = Transformer(vocab_size,args.context_length, args.num_layers, args.d_model, args.num_heads, args.d_ff, 10000.0)
    model = model.to(device)
    model = torch.compile(model)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps = args.num_steps

    for i in range(steps):
        lr = learning_rate_schedule(i, args.lr, args.min_lr, args.warmup_steps, args.num_steps)
        for group in opt.param_groups:
            group["lr"] = lr

        x, targets = get_batch(tokens, args.batch_size, args.context_length, device="cpu")

        opt.zero_grad()

        logits = model(x)
        loss = cross_entropy(logits, targets)
        wandb.log({"train_loss": loss, "step": i})

        if i % 50 == 0:
            print("checkpoint_path:", repr(args.checkpoint_path), type(args.checkpoint_path))
            save_checkpoint(model, opt, i, args.checkpoint_path)

        if i % 10 == 0:
            val_loss = evaluate(model, val_tokens, args.batch_size, args.context_length, device="cpu")
            wandb.log({"val_loss": val_loss, "step": i})

            
            print(f"Validation loss: {val_loss}")
            print(f"Step {i} Training Loss: {loss.item()}")
        loss.backward()

        gradient_clipping(model.parameters(), max_l2_norm=1.0)

        opt.step()

@torch.no_grad()
def evaluate(model, val_tokens, batch_size, context_length, device, eval_iters=20):
    losses = []
    for i in range(eval_iters):
        x, targets = get_batch(val_tokens, batch_size, context_length, device)

        logits = model(x)
        loss = cross_entropy(logits, targets)
        losses.append(loss)
    return sum(losses) / len(losses)

def main():
    # training tokenizer
    # tokenize("data/tinystories_vocab.pkl", "data/tinystories_merges.pkl", "data/TinyStoriesV2-GPT4-train.txt", "data/tinystories_tokens_train.bin")
    # tokenize("data/tinystories_vocab.pkl", "data/tinystories_merges.pkl", "data/TinyStoriesV2-GPT4-valid.txt", "data/tinystories_tokens_val.bin")

    # training transformer
    args = parse_args()
    wandb.init(project="cs336-transformer", config=vars(args))
    train()
    wandb.finish()


    



if __name__ == "__main__":
    main()