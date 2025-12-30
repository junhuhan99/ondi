"""
OnDi - Custom BPE Tokenizer from Scratch
Trained on Coding & English data
"""

import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import os


class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer
    Custom implementation for OnDi model
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<sep>': 4,
            '<mask>': 5,
            '<code>': 6,
            '</code>': 7,
        }
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def _get_stats(self, vocab: Dict[str, int]) -> Counter:
        """Count frequency of adjacent pairs"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge most frequent pair in vocabulary"""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq

        return new_vocab

    def train(self, texts: List[str], min_frequency: int = 2, verbose: bool = True):
        """Train BPE tokenizer on texts"""
        if verbose:
            print("Training BPE Tokenizer...")
            print(f"Target vocab size: {self.vocab_size}")

        # Tokenize into words and count frequencies
        word_freqs = Counter()
        for text in texts:
            # Split by whitespace and punctuation, keeping code tokens
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            word_freqs.update(words)

        # Filter by minimum frequency
        word_freqs = {word: freq for word, freq in word_freqs.items() if freq >= min_frequency}

        # Initialize vocabulary with characters
        vocab = {}
        for word, freq in word_freqs.items():
            # Add spaces between characters
            vocab[' '.join(list(word)) + ' </w>'] = freq

        # Get unique characters
        chars = set()
        for word in vocab.keys():
            chars.update(word.split())

        # Initialize token vocabulary
        self.vocab = dict(self.special_tokens)
        idx = len(self.special_tokens)

        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = idx
                idx += 1

        # BPE merges
        num_merges = self.vocab_size - len(self.vocab)
        if verbose:
            print(f"Initial vocab size: {len(self.vocab)}")
            print(f"Performing {num_merges} merges...")

        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)

            merged_token = ''.join(best_pair)
            self.merges[best_pair] = merged_token

            if merged_token not in self.vocab:
                self.vocab[merged_token] = idx
                idx += 1

            if verbose and (i + 1) % 1000 == 0:
                print(f"  Merge {i + 1}/{num_merges}")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using learned merges"""
        word = ' '.join(list(word)) + ' </w>'
        tokens = word.split()

        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            mergeable = [(pair, self.merges.get(pair)) for pair in pairs if pair in self.merges]

            if not mergeable:
                break

            # Find the pair that was merged earliest
            best_pair = min(mergeable, key=lambda x: list(self.merges.keys()).index(x[0]) if x[0] in self.merges else float('inf'))
            pair, merged = best_pair

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        words = re.findall(r'\w+|[^\w\s]', text.lower())

        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                tokens.append(self.vocab.get(token, self.unk_token_id))

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        tokens = []
        for tid in token_ids:
            token = self.inverse_vocab.get(tid, '<unk>')
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)

        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

    def save(self, path: str):
        """Save tokenizer to disk"""
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        merges_list = [f"{p[0]} {p[1]}" for p in self.merges.keys()]
        with open(os.path.join(path, 'merges.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(merges_list))

        config = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from disk"""
        with open(os.path.join(path, 'vocab.json'), 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        with open(os.path.join(path, 'merges.txt'), 'r', encoding='utf-8') as f:
            merges_list = f.read().strip().split('\n')

        self.merges = {}
        for merge in merges_list:
            if merge:
                parts = merge.split()
                if len(parts) == 2:
                    self.merges[(parts[0], parts[1])] = parts[0] + parts[1]

        with open(os.path.join(path, 'config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.vocab_size = config['vocab_size']
            self.special_tokens = config['special_tokens']

        print(f"Tokenizer loaded from {path}")

    def __len__(self):
        return len(self.vocab)


if __name__ == '__main__':
    # Test tokenizer
    print("Testing BPE Tokenizer...")

    sample_texts = [
        "def hello_world():",
        "    print('Hello, World!')",
        "function greet(name) {",
        "    return `Hello, ${name}!`;",
        "}",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ]

    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(sample_texts * 100, min_frequency=1, verbose=True)

    print("\nTest encoding/decoding:")
    test_text = "def test_function(): print('hello')"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {test_text}")
    print(f"Encoded: {encoded[:20]}...")
    print(f"Decoded: {decoded}")

    print("\nTokenizer test successful!")
