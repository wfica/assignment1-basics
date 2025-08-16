from tokenizers.models import BPE
import pickle
from cs336_basics.tokenizer_impl import Tokenizer
from cs336_basics.tokenizer_exp import tokenize_and_save
import numpy as np


def main():
    ############
    v1 = np.fromfile("/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-valid.bin", dtype=np.int16)
    print(len(v1))
    # v2 = np.fromfile("/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-valid-2.bin", dtype=np.int16)
    # print(len(v2))
    # for i in range(min(len(v1), len(v2))):
    #     if v1[i] != v2[i]:
    #         print("The tokenizations is different!")
    #         break
    # print("The tokenization is the same!")
    #############
    
    special_tokens = ["<|endoftext|>"]
    tiny_stories_tokenizer = Tokenizer.from_files(
        "/Users/fica/cs336/assignment1-basics/data/tiny_stories_bpe_vocab.pkl",
        "/Users/fica/cs336/assignment1-basics/data/tiny_stories_bpe_merges.pkl",
        special_tokens=special_tokens,
    )

    print(f"Max token in vocab: {max(tiny_stories_tokenizer.vocab.keys())}")
    print(tiny_stories_tokenizer.decode(v1[:1000]))
    return
    # tokenize_and_save(
    #     tiny_stories_tokenizer,
    #     "/Users/fica/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
    #     "/Users/fica/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-tokens_10k.bin",
    # )
    train_ids = np.memmap("/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-train.bin", dtype=np.int16, mode="r")
    print(train_ids[:100])
    
    # with open("/Users/fica/cs336/assignment1-basics/data/tiny_stories_bpe_vocab.pkl", "rb") as f:
    #   vocab = pickle.load(f)
    # assert type(vocab) == dict
    # with open("/Users/fica/cs336/assignment1-basics/data/tiny_stories_bpe_merges.pkl", "rb") as f:
    #     merges = pickle.load(f)
    # assert type(merges) == list
    # for k, v in vocab.items():
    #    print (k, v)
    #    break

    # print(merges[0])

    # tiny_stories_train = sample_documents(
    #     "/Users/fica/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
    #     special_tokens,
    # )
    # tiny_stories_valid = sample_documents(
    #     "/Users/fica/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
    #     special_tokens,
    # )

if __name__ == "__main__":
    main()
