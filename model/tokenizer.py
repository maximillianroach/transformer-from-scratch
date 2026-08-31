from typing import Iterable, Iterator
import pickle
import re 
import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # append special tokens to vocab if they aren't already there
        count = len(vocab)

        if special_tokens:
            for special_token in special_tokens:
                if special_token.encode("utf-8") not in vocab.values():
                    vocab[count] = special_token.encode("utf-8")
                    count += 1

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.bytes_to_ids = {b: i for i, b in self.vocab.items()}

        # maps pairs in merges to their order in the merges list
        self.merge_ranks = {pair: id for id, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        # we have a reference to the class so we use cls to call the class constructor
        return cls(vocab, merges, special_tokens)

    # returns the ids corresponds to the word bytes
    def _encode_word(self, word: bytes):
        new_word = [bytes([b]) for b in word]

        # scan through all pairs in the word
        while len(new_word) > 1:
            pairs = [(new_word[i], new_word[i+1]) for i in range(len(new_word) - 1)]

            # find which pairs are merges
            mergeable_pairs = [p for p in pairs if p in self.merge_ranks]

            # no more mergeable pairs so we're done encoding
            if not mergeable_pairs:
                break

            # choose the pair that occurs earliest in merges (this is why the order of merges is important)
            pair_ids = {p: self.merge_ranks[p] for p in mergeable_pairs}
            best_pair = min(pair_ids, key=pair_ids.get)

            merged = []
            i = 0
            # we must loop through all the letters if the last pair doesn't work since we still need that last character
            while i < len(new_word):
                if i < len(new_word) - 1 and best_pair == (new_word[i], new_word[i+1]):
                    merged.append(b''.join(best_pair))
                    i += 2
                else:
                    merged.append(new_word[i])
                    i += 1
            new_word = merged

        word_encoding = [self.bytes_to_ids[w] for w in new_word]
        return word_encoding

    def encode(self, text: str) -> list[int]:
        # split on special tokens before pretokenizing 

        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            documents = re.split(pattern, text)
        else:
            documents = [text]

        # pre-tokenize with GPT regex
        p = regex.compile(PAT)

        encoding = []

        for doc in documents:
            if self.special_tokens and doc in self.special_tokens:
                encoding.append(self.bytes_to_ids[doc.encode("utf-8")])
                continue

            words = p.findall(doc)
            byte_words = [w.encode("utf-8") for w in words]

            for word in byte_words:
                encoding.extend(self._encode_word(word))

        return encoding
            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            for id in self.encode(line):
                yield id

    def decode(self, ids: list[int]) -> str:
        byte_array = [self.vocab[id] for id in ids]
        byte_sequence = b''.join(byte_array)
        return byte_sequence.decode("utf-8", errors="replace")

