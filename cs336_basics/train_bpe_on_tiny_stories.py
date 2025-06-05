from .tokenizer_impl import train_bpe
# from datasets import load_dataset
import pickle

if __name__ == "__main__":
    # ds = load_dataset("roneneldan/TinyStories")
    # print(ds)
    vocab, merges = train_bpe(input_path="/Users/fica/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",  # Use the dataset directly
              vocab_size=10_000,
              special_tokens=["<|endoftext|>"], num_processes=32)
    
    with open("/Users/fica/cs336/assignment1-basics/data/tiny_stories_bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("/Users/fica/cs336/assignment1-basics/data/tiny_stories_bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    longest_word = max(vocab, key=lambda k: len(vocab[k]))
    print(f"Longest word in the vocabulary: {longest_word} with length {len(vocab[longest_word])}")

    