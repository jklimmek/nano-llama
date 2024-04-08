import argparse
import logging
import gc
import os

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
from utils import set_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory with files to tokenize.")
    parser.add_argument("--out", type=str, help="Where to output tokenized files.")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer.")
    parser.add_argument("--out_prefix", type=str, default="train", help="Output file name prefix.")
    parser.add_argument("--min_length", type=int, default=50, help="All sequences shorter than min_length will be skipped, -1 means all will be kept.")
    parser.add_argument("--chunk_size", type=int, default=1024*4, help="Number of lines to read from file at a time.")
    parser.add_argument("--file_size", type=int, default=128, help="Approximate number of MBs that fill be saved into file.")
    args = parser.parse_args()
    return vars(args)


def tokenize(args):
    """Tokenize a directory of files."""
    set_logging()
    os.makedirs(args["out"], exist_ok=True)

    tokenizer = Tokenizer.from_file(args["tokenizer"])
    eot = tokenizer.token_to_id("[END-OF-TEXT]")
    files = [f for f in os.scandir(args["dir"]) if f.is_file()]

    logging.info(f"Tokenizing {len(files)} files.")

    total_tokens = 0
    file_counter = 0
    current_file_size = 0
    tokenized_lines = []

    for file in tqdm(files, total=len(files), ncols=100):
        for chunk_lines in read_file_in_chunks(file, args["chunk_size"]):
            for line in chunk_lines:
                
                # [END-OF-TEXT] token is append delimitering each article.
                tokens = np.asarray(tokenizer.encode(str(line)).ids + [eot], dtype=np.uint16)

                # Skip sequence if it's too short.
                if len(tokens) < args["min_length"] and args["min_length"] != -1:
                    continue

                tokenized_lines.append(tokens)
                total_tokens += len(tokens)
                current_file_size += tokens.itemsize * tokens.size / 1024 ** 2

                # If file exceeds desired limit save tokens, 
                # increment file counter and clean up.
                if current_file_size > args["file_size"]:

                    tokenized_array = np.concatenate(tokenized_lines)
                    np.save(os.path.join(args["out"], f'{args["out_prefix"]}_{file_counter}'), tokenized_array)
                    file_counter += 1
                    current_file_size = 0
                    tokenized_lines = []
                    gc.collect()
    
    # If file was too small to divide into more files, save it as one.
    if file_counter == 0 or len(tokenized_lines) > 0:
        tokenized_array = np.concatenate(tokenized_lines)
        np.save(os.path.join(args["out"], f'{args["out_prefix"]}_{file_counter}'), tokenized_array)
        file_counter += 1

    logging.info(f"Total of {total_tokens:,} tokens in {file_counter} file(s).")


def read_file_in_chunks(filename, chunk_size):
    """Reads file in chunks."""
    with open(filename, 'r', encoding="utf-8") as file:
        while True:
            chunk = []
            for _ in range(chunk_size):
                line = file.readline()
                if not line:
                    break
                chunk.append(line)
            if not chunk:
                break
            yield chunk


def main():
    args = parse_args()
    tokenize(args)


if __name__ == "__main__":
    main()