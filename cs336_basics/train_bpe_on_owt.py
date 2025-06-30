from cs336_basics.tokenizer_impl import train_bpe
import pickle
import time

if __name__ == "__main__":
    start_time = time.time()
    num_processes= 32
    vocab, merges = train_bpe(input_path="/Users/fica/cs336/assignment1-basics/data/owt_train.txt",  # Use the dataset directly
              vocab_size=32_000,
              special_tokens=["<|endoftext|>"], num_processes=num_processes)
    end_time = time.time()
    print(f"Training BPE took {end_time - start_time:.2f} seconds", flush=True)

    with open("/Users/fica/cs336/assignment1-basics/data/owt_bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("/Users/fica/cs336/assignment1-basics/data/owt_bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    longest_word = max(vocab, key=lambda k: len(vocab[k]))
    print(f"Longest word in the vocabulary: {vocab[longest_word]} with length {len(vocab[longest_word])}")

    