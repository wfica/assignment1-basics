#!/bin/bash
# Example commands to run the cs336_basics.training_script module

# Basic training run with default parameters
# uv run -m cs336_basics.training_script \
#   --train_data "/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-train.bin" \
#   --val_data "/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-train.valid"

# # Training with custom batch size and context length
# uv run -m cs336_basics.training_script \
#   --train_data "/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-train.bin" \
#   --val_data "/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-train.valid" \
#   --batch_size 64 \
#   --context_length 512

# # Training on CUDA device with float16
# uv run -m cs336_basics.training_script \
#   --train_data "/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-train.bin" \
#   --val_data "/Users/fica/cs336/assignment1-basics/data/tiny_stories_tokens-train.valid" \
#   --device "cuda:0" \
#   --dtype "float16"

############################################### 
# Overfit test: always train on the same batch
# uv run -m cs336_basics.training_script \
#   --device="mps:0" \
#   --batch_size 64 \
#   --context_length 256 \
#   --training_steps 150 \
#   --val_every 10 \
#   --out_dir="/Users/fica/cs336/assignment1-basics/data/training_1"

# uv run -m cs336_basics.training_script \
#   --device="mps:0" \
#   --batch_size 64 \
#   --context_length 256 \
#   --training_steps 2000 \
#   --val_every 100 \
#   --save_ckpt_every 250 \
#   --out_dir="/Users/fica/cs336/assignment1-basics/data/training_2"


# uv run -m cs336_basics.training_script \
#   --device="mps:0" \
#   --batch_size 64 \
#   --context_length 256 \
#   --training_steps 2000 \
#   --val_every 100 \
#   --save_ckpt_every 250 \
#   --adamw_weight_decay 0.05 \
#   --max_lr 0.0002 \
#   --out_dir="/Users/fica/cs336/assignment1-basics/data/training_3"

uv run -m cs336_basics.training_script \
  --device="mps:0" \
  --batch_size 64 \
  --context_length 256 \
  --training_steps 2000 \
  --val_every 50 \
  --save_ckpt_every 250 \
  --adamw_weight_decay 0.05 \
  --max_lr 0.0001 \
  --out_dir="/Users/fica/cs336/assignment1-basics/data/training_4"

uv run -m cs336_basics.training_script \
  --device="mps:0" \
  --batch_size 64 \
  --context_length 256 \
  --training_steps 2000 \
  --val_every 50 \
  --save_ckpt_every 250 \
  --adamw_weight_decay 0.05 \
  --max_lr 0.00005 \
  --out_dir="/Users/fica/cs336/assignment1-basics/data/training_5"

uv run -m cs336_basics.training_script \
  --device="mps:0" \
  --batch_size 64 \
  --context_length 256 \
  --training_steps 2000 \
  --val_every 50 \
  --save_ckpt_every 250 \
  --adamw_weight_decay 0.75 \
  --max_lr 0.00005 \
  --out_dir="/Users/fica/cs336/assignment1-basics/data/training_6"