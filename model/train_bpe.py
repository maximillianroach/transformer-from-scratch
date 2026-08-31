import re
import regex
from .pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
from typing import BinaryIO
from collections import Counter, defaultdict

# GPT Tokenizer regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    str_chunk = chunk.decode("utf-8", errors="ignore")

    documents = re.split(pattern, str_chunk)

    p = regex.compile(PAT)

    c = Counter()
    for doc in documents:
        pretokens = []
        for tok in p.finditer(doc):
            pretokens.append(tok.group())
        c.update(pretokens)

    return c

def pretokenize(input_path: str, special_tokens: list[str]):
    with open(input_path, "rb") as f:
        encoded_special_token = special_tokens[0].encode('utf-8')
        num_processes = 4
        chunk_boundaries = find_chunk_boundaries(f, num_processes, encoded_special_token)

        # supplied to each worker in paralle
        chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]

        # Parallelize pretokenizing the chunks
        with Pool(processes=num_processes) as pool:
            all_counters = pool.map(pretokenize_chunk, chunk_args)


        total_counter = Counter()
        for counter in all_counters:
            total_counter.update(counter)
    return total_counter

# the pretokenizer uses strings and outputs the frequency dict with string keys, but the tokenizer expects tuples
def pretoken_counter_to_tuples(counter):
    vocab = {}

    for pretoken, freq in counter.items():
        symbols = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
        vocab[symbols] = freq
    return vocab

# a one-time calculation to get the frequency of each pair
def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():

        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += freq
    return pairs

# this is for all the overall vocabular so we adjust by the frequency of the word
def merge_word(word: tuple, pair: tuple, pairs: dict, word_freq: int):
    new_word = []

    pair_block = b''.join(pair)

    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
            # this gives us correct answer when multiple occurrences of pair are next to each other
            left = -1 if new_word else None
            right = i + 2 if i + 2 < len(word) else None

            # the new frequency with the pair becomes the old frequency between the left character and the left character of pair
            if left is not None:
                pairs[(new_word[left], pair_block)] += word_freq
                pairs[(new_word[left], pair[0])] -= word_freq

            # the new frequency with the pair becomes the old frequency between the right character and the right character of pair
            if right is not None:
                pairs[(pair_block, word[right])] += word_freq
                pairs[(pair[1], word[right])] -= word_freq

            # set the freq for the characters in the pair to 0
            pairs[(pair[0], pair[1])] -= word_freq

            new_word.append(pair_block)
            # we skip both of the characters we just replaced
            i += 2

        # The pair doesn't match the desired pair
        else:
            new_word.append(word[i])
            i += 1

    return tuple(new_word)

def merge_vocab(pair, pairs, v_in):
    v_out = {}
    
    for word in v_in:
        new_word = merge_word(word, pair, pairs, v_in[word])
        # we need to add like this because during merging, two words may converge on the same new word so we need to sum
        # those frequencies 
        v_out[new_word] = v_out.get(new_word, 0) + v_in[word]

    return v_out

def train_bpe(input_path, vocab_size, special_tokens):
    counter = pretokenize(input_path, special_tokens)
    vocab = pretoken_counter_to_tuples(counter)

    pairs = get_stats(vocab)

    merges = []

    vocab_symbols = [bytes([b]) for b in range(256)]

    num_merges = vocab_size - 256 - len(special_tokens)
    for j in range(num_merges):
        best = max(pairs, key=lambda p: (pairs[p], p))

        # record the pair that we merged this step
        merges.append(best)

        vocab_symbols.append(b''.join(best))

        vocab = merge_vocab(best, pairs, vocab)

        # we don't want this pair anymore since it's been merged
        del pairs[best]

    trained_vocab = {i: token for i, token in enumerate(vocab_symbols + [tok.encode('utf-8') for tok in special_tokens])}

    return trained_vocab, merges











    

        


