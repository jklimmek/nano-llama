import argparse
import os

from tokenizers.implementations import SentencePieceBPETokenizer
from tokenizers import pre_tokenizers, normalizers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory with files to train tokenizer.")
    parser.add_argument("--out", type=str, help="Where to output trained tokenizer.")
    parser.add_argument("--vocab_size", type=int, help="Vocabulary size of tokenizer.")
    args = parser.parse_args()
    return vars(args)


def train_and_save(args):
    """Trains and saves tokenizer."""

    initial_alphabet = list("0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\]_|~")

    replacements = {
        '"': '',
        "‘": "'",
        "’": "'",
        " ": "",
        "﻿": "",
        "“": '"',
        "”": '"',
        "–": '-'
    }

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.normalizer = normalizers.Sequence(
        [
            *[normalizers.Replace(k, v) for k, v in replacements.items()],
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Strip(),
        ]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Metaspace(),
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.Punctuation(behavior="isolated"),
        ]
    )

    train_files = [os.path.join(root, file) for root, _, files in os.walk(args["dir"]) for file in files]

    tokenizer.train(
        files=train_files,
        vocab_size=args["vocab_size"],
        limit_alphabet=len(initial_alphabet) + 1,
        initial_alphabet=initial_alphabet,
        special_tokens=["[PADDING]", "[END-OF-TEXT]", "[TITLE]", "[CONTEXT]"]
    )

    directory = os.path.dirname(args["out"])
    os.makedirs(directory, exist_ok=True)
    tokenizer.save(args["out"])


def main():
    args = parse_args()
    train_and_save(args)


if __name__ == "__main__":
    main()