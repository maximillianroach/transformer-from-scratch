from .decoding import decode
from .tokenizer import Tokenizer
from .transformer import Transformer
from .decoding import decode
from .checkpointing import load_checkpoint
from .adamw import AdamW
import torch

def main():
    tokenizer = Tokenizer.from_files("data/tinystories_vocab.pkl", "data/tinystories_merges.pkl", special_tokens=["<|endoftext|>"])
    vocab_size = len(tokenizer.vocab)

    model = Transformer(10000, 256, 4, 512, 16, 1344, 10000)

    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1)

    load_checkpoint("checkpoint.pth", model, opt)

    prompt = "Once upon a time,"

    prompt_tokens = torch.tensor(tokenizer.encode(prompt))
    prompt_tokens_len = prompt_tokens.shape[0]
    prompt_tokens = prompt_tokens.reshape((prompt_tokens_len, -1)).unsqueeze(1)

    eos_token = tokenizer.encode("<|endoftext|>")[0]

    output_tokens = decode(model=model, prompt=prompt_tokens, max_tokens=500, temperature=0.8, top_p=0.8, eos_token_id=eos_token)

    out_text = tokenizer.decode(output_tokens)

    print(out_text)
    print(tokenizer.decode([60]))

if __name__ == "__main__":
    main()