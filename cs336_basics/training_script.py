import torch
import numpy as np
from cs336_basics.training_impl import training_loop
import argparse
import os
from cs336_basics.tokenizer_impl import Tokenizer
import json


def get_tokenizer():
    special_tokens = ["<|endoftext|>"]
    tiny_stories_tokenizer = Tokenizer.from_files(
        "/Users/fica/cs336/assignment1-basics/data/tiny_stories_bpe_vocab.pkl",
        "/Users/fica/cs336/assignment1-basics/data/tiny_stories_bpe_merges.pkl",
        special_tokens=special_tokens,
    )
    return tiny_stories_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--always_train_on_the_same_batch",
        type=lambda x: x.lower() == "true",
        required=False,
        default=False,
        help="Used to see if we can overfit quickly.",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-train.bin",
        help="Path to training .npy or .bin file",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-valid.bin",
        help="Path to validation .npy or .bin file",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--training_steps", type=int, default=5_000)
    parser.add_argument("--save_ckpt_every", type=int, default=1_000)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/Users/fica/cs336/assignment1-basics/data/training",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument(
        "--val_every", type=int, default=500, help="Validate every N steps"
    )
    parser.add_argument(
        "--max_lr", type=float, default=3e-4, help="Maximum learning rate for scheduler"
    )
    parser.add_argument(
        "--min_lr", type=float, default=3e-5, help="Minimum learning rate for scheduler"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for scheduler",
    )
    parser.add_argument(
        "--gradient_clipping_max_l2_norm",
        type=float,
        default=None,
        help="Max L2 norm for gradient clipping",
    )
    parser.add_argument(
        "--adamw_lr", type=float, default=0.001, help="AdamW learning rate"
    )
    parser.add_argument(
        "--adamw_betas",
        type=lambda s: tuple(map(float, s.split(","))),
        default="0.9,0.999",
        help="AdamW betas, comma separated",
    )
    parser.add_argument(
        "--adamw_weight_decay", type=float, default=0.01, help="AdamW weight decay"
    )
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS (Appe metal chip) backend not available.")
    
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(
            vars(args)
            | {"torch.float32_matmul_precision": torch.get_float32_matmul_precision()},
            f,
            indent=4,
        )

    # Memory-efficient loading
    train_ids = np.memmap(args.train_data, dtype=np.int16, mode="r")
    val_ids = np.memmap(args.val_data, dtype=np.int16, mode="r")
    if len(train_ids) == len(val_ids) and all(
        map(lambda p: p[0] == p[1], zip(train_ids[:100], val_ids[:100]))
    ):
        print("Training and validation data seems to be the same.")

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    # Run training loop
    losses_tv = training_loop(
        training_steps=args.training_steps,
        save_ckpt_every=args.save_ckpt_every,
        array_with_training_text_tokens=train_ids,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        out_dir=args.out_dir,
        device=args.device,
        dtype=dtype,
        val_ids=val_ids,
        val_every=args.val_every,
        always_train_on_the_same_batch=args.always_train_on_the_same_batch,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        gradient_clipping_max_l2_norm=args.gradient_clipping_max_l2_norm,
        adamw_lr=args.adamw_lr,
        adamw_betas=args.adamw_betas,
        adamw_weight_decay=args.adamw_weight_decay,
    )


if __name__ == "__main__":
    main()
