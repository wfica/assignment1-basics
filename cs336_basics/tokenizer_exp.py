import pickle
from typing import Any
from .tokenizer_impl import Tokenizer, find_chunk_boundaries
import time
import multiprocessing
import numpy as np
from tqdm import tqdm 
from tqdm.contrib.concurrent import process_map 

def read_vocab_and_meerges(vocab_fp: str, merges_fp: str) -> tuple[Any]:
    with open(vocab_fp, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_fp, "wb") as f:
        merges = pickle.load(f)
    return vocab, merges


def sample_documents(
    fp: str, special_tokens: list[str], max_num_sample: int = 10, chunk_size: int = 4096
) -> str:
    assert len(special_tokens) == 1
    special_token = special_tokens[0]
    content = ""
    count = 0
    with open(fp, "r") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            content += chunk
            count += chunk.count(special_token)
            if count >= max_num_sample:
                content = (
                    special_token.join(content.split(special_token)[:max_num_sample])
                    + special_token
                )
                break
    return content


def tokenize_and_print_stats(tokenizer: Tokenizer, docs: str) -> None:
    encoding = tokenizer.encode(docs)
    assert tokenizer.decode(encoding) == docs
    print(f"Encoding: {encoding[:10]} ...")
    print(f"Compression ratio (bytes/token): {len(docs.encode("utf-8"))/len(encoding)}")
 

def task_a(tiny_stories_tokenizer: Tokenizer, tiny_stories_docs: str,owt_tokenizer:Tokenizer,owt_docs :str):
    print(f"********  TASK A  ********\n  - Tiny Stories:")
    tokenize_and_print_stats(tiny_stories_tokenizer, tiny_stories_docs)
    print("  - OWT:")
    tokenize_and_print_stats(owt_tokenizer, owt_docs)


def task_b(tiny_stories_tokenizer: Tokenizer, owt_docs :str):
    print(f"********  TASK B  ********")
    tokenize_and_print_stats(tiny_stories_tokenizer, owt_docs)


def task_c(tokenizer: Tokenizer, long_text: str):
    print(f"********  TASK C  ********")
    start = time.time()
    _ = tokenizer.encode(long_text)
    end = time.time()
    total_seconds = end - start
    total_bytes = len(long_text.encode("utf-8"))
    pps = total_bytes / total_seconds
    print(f"It took {total_seconds} to tokenize a text of {total_bytes} bytes.")
    total_pile_text_time = 825_000_000_000 / pps / 60 / 60 / 24
    print(f"Id'd take {total_pile_text_time} days to tokenize the Pile dataset")
    total_owt_time = 11_000_000_000 / pps / 60 / 60 / 24
    print(f"Id'd take {total_owt_time} days to tokenize the OWT dataset")


def tokenize_and_save(tokenizer: Tokenizer, in_fp: str, out_fp: str, num_processes = 1, special_tokens = ["<|endoftext|>"]):
    assert len(special_tokens) == 1
    special_token = special_tokens[0]
    with open(in_fp, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes*10_000, special_token.encode("utf-8"))
        print(len(boundaries))
        chunks = []
        for start, end in tqdm(zip(boundaries[:-1], boundaries[1:])):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
        results = process_map(tokenizer.encode, chunks, chunksize=4)

    encoding = [token for chunk_encoding in results for token in chunk_encoding]
    print(f"Encoded {in_fp} in {len(encoding)} tokens.")
    np.array(encoding, dtype=np.uint16).tofile(out_fp)


def task_d(tiny_stories_tokenizer,owt_tokenizer ):
    print(f"********  TASK D  ********")
    tokenize_and_save(tiny_stories_tokenizer, 
                    "/Users/fica/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
                    "/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-valid-2.bin")  
    # tokenize_and_save(tiny_stories_tokenizer, 
    #                 "/Users/fica/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
    #                 "/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-train.bin")


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    tiny_stories_tokenizer = Tokenizer.from_files(
        "/Users/fica/cs336/assignment1-basics/data/tiny_stories_bpe_vocab.pkl",
        "/Users/fica/cs336/assignment1-basics/data/tiny_stories_bpe_merges.pkl",
        special_tokens=special_tokens,
    )
    tiny_stories_docs = sample_documents(
        "/Users/fica/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        special_tokens,
    )
    owt_tokenizer = Tokenizer.from_files(
        "/Users/fica/cs336/assignment1-basics/data/owt_bpe_vocab.pkl",
        "/Users/fica/cs336/assignment1-basics/data/owt_bpe_merges.pkl",
        special_tokens=special_tokens,
    )
    owt_docs = sample_documents(
        "/Users/fica/cs336/assignment1-basics/data/owt_train.txt",
        special_tokens=special_tokens,
    )
    task_a(tiny_stories_tokenizer, tiny_stories_docs, owt_tokenizer, owt_docs)
    task_b(tiny_stories_tokenizer, owt_docs)
    long_text = sample_documents(
        "/Users/fica/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        special_tokens,
        max_num_sample=100
    )
    task_c(tiny_stories_tokenizer, long_text)
    # task_d(tiny_stories_tokenizer, owt_tokenizer)


# ********  TASK A  ********
#   - Tiny Stories:
# Encoding: [10, 430, 439, 259, 398, 401, 283, 259, 390, 496] ...
# Compression ratio (bytes/token): 4.161166116611661
#   - OWT:
# Encoding: [2006, 3390, 696, 361, 473, 284, 3899, 2052, 361, 1885] ...
# Compression ratio (bytes/token): 4.703510859863136
# ********  TASK B  ********
# Encoding: [1071, 2717, 492, 349, 375, 266, 2339, 1679, 349, 867] ...
# Compression ratio (bytes/token): 3.2030189443825345
# ********  TASK C  ********
# It took 12.016009092330933 to tokenize a text of 80600 bytes.
# Id'd take 1423.5260289112068 days to tokenize the Pile dataset
# Id'd take 18.980347052149426 days to tokenize the OWT dataset