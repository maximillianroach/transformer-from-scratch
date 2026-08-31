from .train_bpe import train_bpe
import time
import resource
import pickle

def main():
    t0 = time.time()
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", vocab_size=10000, special_tokens=["<|endoftext|>"])
    elapsed = time.time() - t0

    with open("tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    longest_string_index = max(vocab, key=lambda x: len(vocab[x]))
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"Peak memory usage: {peak_mb:.2f} MB")
    print(f'Time Elapsed: {elapsed}')
    print(f'Longest string (bytes): {max(vocab.values(), key=len)}')
    print(f'Longest string: {vocab[longest_string_index].decode("utf-8")}')

if __name__ == "__main__":
    main()

