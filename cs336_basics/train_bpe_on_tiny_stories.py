from .tokenizer_impl import train_bpe
# from datasets import load_dataset
import pickle
import time

if __name__ == "__main__":
    # ds = load_dataset("roneneldan/TinyStories")
    # print(ds)
    num_processes = 32  # Adjust this based on your system's capabilities
    print(f"Training BPE with {num_processes} processes", flush=True)

    start_time = time.time()
    vocab, merges = train_bpe(input_path="/Users/fica/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",  # Use the dataset directly
            vocab_size=10_000,
            special_tokens=["<|endoftext|>"], num_processes=num_processes)
    end_time = time.time()
    print(f"Training BPE took {end_time - start_time:.2f} seconds", flush=True)
    
    # Save the vocabulary and merges to files
    # with open("/Users/fica/cs336/assignment1-basics/data/tiny_stories_bpe_vocab.pkl", "wb") as f:
    #     pickle.dump(vocab, f)
    # with open("/Users/fica/cs336/assignment1-basics/data/tiny_stories_bpe_merges.pkl", "wb") as f:
    #     pickle.dump(merges, f)

    longest_word = max(vocab, key=lambda k: len(vocab[k]))
    print(f"Longest word in the vocabulary: {vocab[longest_word]} with length {len(vocab[longest_word])}")

    