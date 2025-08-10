import torch
import numpy as np
from cs336_basics.training_impl import training_loop
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True, help='Path to training .npy or .bin file')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation .npy or .bin file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--context_length', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--rope_theta', type=float, default=10000.0)
    parser.add_argument('--training_steps', type=int, default=10000)
    parser.add_argument('--save_ckpt_every', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--val_every', type=int, default=500, help='Validate every N steps')
    args = parser.parse_args()

    # Memory-efficient loading
    train_ids = np.memmap(args.train_data, dtype=np.int32, mode='r')
    val_ids = np.memmap(args.val_data, dtype=np.int32, mode='r')

    # Convert dtype string to torch dtype
    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
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
    )
    np.save(os.path.join(args.out_dir, "losses_train"), losses_tv[0])
    np.save(os.path.join(args.out_dir, "losses_valid"), losses_tv[1])
    
if __name__ == "__main__":
    main()