import torch
import numpy as np
from cs336_basics.decoding import decode
import argparse
import os
from cs336_basics.tokenizer_impl import Tokenizer
import json
from cs336_basics.transformer_impl import Transformer
from cs336_basics.tokenizer_impl import Tokenizer
from cs336_basics.training_impl import find_latest_checkpoint
from cs336_basics.training_script import get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir",
        type=str,
        default="/Users/fica/cs336/assignment1-basics/data/training_Sep_14_2",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="Once upon a time there was a little boy named Ben. Ben ",
    )
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS (Appe metal chip) backend not available.")

    with open(os.path.join(args.train_dir, "args.json"), "r") as f:
        train_args = json.load(f)

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    tokenizer = get_tokenizer()
    assert len(tokenizer.encode("<|endoftext|>")) == 1
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]

    model = Transformer(
        train_args["vocab_size"],
        train_args["context_length"],
        train_args["num_layers"],
        train_args["d_model"],
        train_args["num_heads"],
        train_args["d_ff"],
        train_args["rope_theta"],
        train_args["device"],
        dtype_map.get(train_args["dtype"], torch.float32),
    )
    lastest_ckpt, _ = find_latest_checkpoint(train_args["out_dir"])
    model.load_state_dict(torch.load(lastest_ckpt)["model"])
    print("MODEL LOADED.")

    rollout_tokens = decode(
        torch.tensor(tokenizer.encode(args.prefix)),
        model,
        max_decoding_steps=100,
        eos_token_id=eos_token_id,
        temperature=0.5,
    )
    rollout = tokenizer.decode(rollout_tokens.to("cpu").numpy().tolist())
    print(f"{args.prefix} [...] {rollout}")


if __name__ == "__main__":
    main()
