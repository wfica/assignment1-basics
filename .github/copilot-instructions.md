# Copilot instructions for assignment1-basics

Purpose: quick, actionable notes to help an AI coding agent be productive in this repo.

- Start points
  - Read `README.md` first. It documents the environment tool `uv`, test commands, and data download steps.
  - Primary package: `cs336_basics/`. Key files: `tokenizer_impl.py`, `training_impl.py`, `transformer_impl.py`, `decoding.py`, and `training_script.py`.

- Big-picture architecture
  - Tokenization pipeline: `cs336_basics/tokenizer_impl.py` implements a simple BPE trainer and `Tokenizer` class (vocab: int->bytes, merges: list of byte pairs). Pre-tokenization uses `pre_tokenize` and the special token `"<|endoftext|>"`.
  - Training loop and utilities: `cs336_basics/training_impl.py` contains the training loop, optimizer implementations (custom `AdamW`, `SGD`), lr schedule, data loading (`data_loading`), checkpoint save/load, and plotting helpers.
  - Model: `cs336_basics/transformer_impl.py` defines the `Transformer` used by the training loop. `decoding.py` holds the generation routine used during validation.
  - CLI: `cs336_basics/training_script.py` wires tokenizer loading, arg parsing, and calls `training_loop` (it uses absolute example paths in the repo; prefer relative/arg paths when making changes).

- Data & integration points
  - Tokenized training/validation arrays live under `data/tokenization/` as `.bin` files. The training script memory-maps these as `np.int16`.
  - Vocabulary and merges used by the `Tokenizer` are pickled at `data/tokenization/*_bpe_vocab.pkl` and `*_bpe_merges.pkl` and loaded with `Tokenizer.from_files()`.
  - Checkpoints: saved at `out_dir/iter_{i}` (torch.save of a dict). Training also writes `losses_train.npy`, `losses_valid.npy`, and `learning_rates.npy` to `out_dir`.

- Workflows / commands
  - Use `uv` to run code in the managed environment: `uv run cs336_basics/training_script.py --device mps --training_steps 1000`.
  - Run tests with: `uv run pytest` (see `pyproject.toml` for dependencies and pytest settings).
  - Data download: README contains wget commands and there is `data/download.sh` for convenience.

- Project-specific conventions & gotchas
  - Tests initially expect students to implement functions and will fail with `NotImplementedError` until adapters are completed (`tests/adapters.py`).
  - Token IDs are small ints (vocab length typically <= 10000). `data_loading` asserts token IDs are within `vocab_size`.
  - `training_loop` expects exactly one special token (`special_tokens` length == 1) and reads `special_token_to_index["<|endoftext|>"]`.
  - Training uses `torch.compile(..., backend='aot_eager')` and caps lr with `lr = min(lr, 0.00006)` — be careful when tuning schedulers.
  - Device handling: `training_script.py` defaults to `--device mps` and explicitly checks `torch.backends.mps.is_available()` (mac MPS-specific). Torch version constraints are in `pyproject.toml`.

- Code patterns to follow when editing
  - When adding IO or long-running code, follow existing memmap/streaming patterns (e.g., `Tokenizer.encode_iterable` and `np.memmap` usage) to avoid large memory spikes.
  - Checkpointing/restore flow: `find_latest_checkpoint` -> `load_checkpoint` -> expect `losses` length == last_iter. Keep these files compatible.
  - Tests use fixtures in `tests/fixtures/` and helper adapters in `tests/adapters.py` — update adapters to wire new implementations.

- Where to look for examples
  - Tokenizer usage: `cs336_basics/training_script.py::get_tokenizer()` shows `Tokenizer.from_files(...)` with example paths.
  - Training loop usage and checkpointing: `cs336_basics/training_impl.py::training_loop()`.
  - Tests that show expected behaviour: `tests/test_tokenizer.py` and other tests in `tests/`.

If anything above is unclear or you'd like me to expand a section (examples, common PR changes, or tests to add), tell me which part and I'll iterate.
