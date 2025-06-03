from .tokenizer_impl import train_bpe
# from datasets import load_dataset


if __name__ == "__main__":
    # ds = load_dataset("roneneldan/TinyStories")
    # print(ds)
    train_bpe(input_path="/Users/fica/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",  # Use the dataset directly
              vocab_size=10_000,
              special_tokens=["<|endoftext|>"])
    