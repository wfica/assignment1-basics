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

# Overfit test: always train on the same batch
uv run -m cs336_basics.training_script \
  --always_train_on_the_same_batch False \
  --device=mps \
  --batch_size 32 \
  --context_length 256 \
  --training_steps 200 \
  --val_every 10
