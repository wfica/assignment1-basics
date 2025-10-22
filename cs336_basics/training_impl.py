import re
import torch
from torch import nn
from collections.abc import Callable, Iterable
from typing import Optional, BinaryIO, IO
import os
import math
import numpy as np
from einops import rearrange, einsum
import matplotlib.pyplot as plt
import random
from cs336_basics.transformer_impl import Transformer
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from cs336_basics.decoding import decode
from cs336_basics.tokenizer_impl import Tokenizer
from collections import Counter

def plot_training_statistics(
    gradient_stats: list,
    activation_stats: list,
    training_steps: int,
    out_dir: str | os.PathLike,
    output_file_path="training_statistics.png",
):
    """
    Plots gradient and activation statistics over training steps.
    
    Args:
        gradient_stats (list): List of dictionaries containing gradient statistics per step
        activation_stats (list): List of dictionaries containing activation statistics per step
        training_steps (int): Total number of training steps
        out_dir (str | os.PathLike): Directory to save the plot
        output_file_path (str): Name of the output file
    """
    if not gradient_stats or not activation_stats:
        print("Warning: Statistics lists are empty. Nothing to plot.")
        return

    steps = np.arange(1, len(gradient_stats) + 1)
    
    # Create subplots for different statistics
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # Plot gradient statistics
    grad_metrics = ['grad_mean', 'grad_std', 'grad_max', 'grad_min']
    for metric in grad_metrics:
        values = [stats[metric] for stats in gradient_stats]
        ax1.plot(steps, values, label=metric, alpha=0.8)
    
    ax1.set_xlabel("Training Steps", fontsize=12)
    ax1.set_ylabel("Gradient Statistics", fontsize=12)
    ax1.set_title("Gradient Statistics Over Training", fontsize=16, weight="bold")
    ax1.legend(loc="best", fontsize=11)
    ax1.set_yscale('log')  # Use log scale for better visualization
    
    # Plot activation statistics
    # Get all unique parameter names from the first stats entry
    if activation_stats and activation_stats[0]:
        param_names = list(activation_stats[0].keys())
        for param in param_names:
            if '_norm' in param:  # Only plot norms to avoid cluttering
                values = [stats[param] for stats in activation_stats]
                ax2.plot(steps, values, label=param, alpha=0.8)
    
    ax2.set_xlabel("Training Steps", fontsize=12)
    ax2.set_ylabel("Parameter Norms", fontsize=12)
    ax2.set_title("Parameter Norms Over Training", fontsize=16, weight="bold")
    ax2.legend(loc="best", fontsize=11)
    ax2.set_yscale('log')  # Use log scale for better visualization
    
    plt.tight_layout()
    
    # Save the figure
    try:
        plt.savefig(
            os.path.join(out_dir, output_file_path), dpi=300, bbox_inches="tight"
        )
        print(f"Statistics chart saved to {os.path.join(out_dir, output_file_path)}")
    except Exception as e:
        print(f"Error saving the plot: {e}")
    finally:
        plt.close(fig)

def plot_losses_and_learning_rates(
    train_losses,
    val_losses,
    learning_rates,
    val_every,
    out_dir: str | os.PathLike,
    output_file_path="training_plot.png",
):
    """
    Plots training/validation losses and learning rates, and saves the chart to a file.

    Args:
        train_losses (list or np.array): A list of loss values for each training step.
        val_losses (list or np.array): A list of loss values for each validation step.
        learning_rates (list or np.array): A list of learning rate values for each training step.
        val_every (int): The frequency (in steps) at which validation loss was computed.
        out_dir (str or os.PathLike): The directory where the output plot will be saved.
        output_file_path (str): The path where the output plot image will be saved.
    """
    if not train_losses:
        print("Warning: train_losses list is empty. Nothing to plot.")
        return

    training_steps = len(train_losses)

    # Generate x-axis values for the training losses and learning rates
    train_steps_x = np.arange(1, training_steps + 1)

    # Generate x-axis values for the validation losses
    val_steps_x = []
    for i in range(1, training_steps + 1):
        if i % val_every == 0:
            val_steps_x.append(i)
    if training_steps not in val_steps_x:
        val_steps_x.append(training_steps)

    if len(val_losses) != len(val_steps_x):
        print(
            f"Warning: Mismatch between number of validation losses ({len(val_losses)}) "
            f"and calculated validation steps ({len(val_steps_x)}). "
            "Please check your 'val_every' and data."
        )
        min_len = min(len(val_losses), len(val_steps_x))
        val_losses = val_losses[:min_len]
        val_steps_x = val_steps_x[:min_len]

    # Create the plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot training and validation loss on the first y-axis
    p1 = ax1.plot(
        train_steps_x,
        train_losses,
        label="Training Loss",
        color="dodgerblue",
        alpha=0.8,
    )
    p2 = ax1.plot(
        val_steps_x,
        val_losses,
        label="Validation Loss",
        color="darkorange",
        marker="o",
        linestyle="--",
    )

    # Set labels for the first y-axis
    ax1.set_xlabel("Training Steps", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Create a second y-axis for the learning rate that shares the same x-axis. [1]
    ax2 = ax1.twinx()
    p3 = ax2.plot(
        train_steps_x,
        learning_rates,
        label="Learning Rate",
        color="green",
        alpha=0.6,
        linestyle="-.",
    )

    # Set labels for the second y-axis
    ax2.set_ylabel("Learning Rate", fontsize=12, color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Set plot title and a combined legend for all plots
    ax1.set_title(
        "Training & Validation Losses and Learning Rate", fontsize=16, weight="bold"
    )
    plots = p1 + p2 + p3
    labels = [p.get_label() for p in plots]
    ax1.legend(plots, labels, loc="best", fontsize=11)

    # Improve tick readability
    plt.xticks(fontsize=10)

    # Save the figure
    try:
        plt.savefig(
            os.path.join(out_dir, output_file_path), dpi=300, bbox_inches="tight"
        )
        print(f"Chart successfully saved to {os.path.join(out_dir, output_file_path)}")
    except Exception as e:
        print(f"Error saving the plot: {e}")
    finally:
        plt.close(fig)

    # Display the plot
    # plt.show()


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """A function to compute the cross entropy loss, which takes in predicted logits
    (o_i) and targets (x_{i+1}) and computes the cross entropy l_i =-log softmax(o_i)[x_{i+1}].
    Args:
     * logits: input float tensor of shape (..., voc_len)
     * targets: input int tensor of shape (...) - same batch dimentions as logits
    The method
     * Subtract the largest element for numerical stability.
     * Cancel out log and exp whenever possible.
     * Handle any additional batch dimensions and return the average across the batch.
    We assume batch-like dimensions always come first, before the vocabulary size dimension.
    """
    assert logits.shape[:-1] == targets.shape
    logits_scaled = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    e_logits_scaled = torch.exp(logits_scaled)
    nll = -torch.gather(logits_scaled, dim=-1, index=targets.unsqueeze(-1)).squeeze(
        -1
    ) + torch.log(torch.sum(e_logits_scaled, dim=-1))
    assert nll.shape == targets.shape
    return nll.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}.")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8
    ):
        """Args:
        * params - model parameteres (might be parameter group)
        * lr: default learning rate
        * betas: tuple with (scaling of gradient estimate, scaling of gradient square estimate)
        * weight_decay: scaling for weight decay
        * eps: for numerical stability"""
        defaults = {
            "lr": lr,
            "b1": betas[0],
            "b2": betas[1],
            "wd": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # Get the hyperparameters, if not specified for a group fallback to the defaults
            lr, b1, b2, wd, eps = (
                group["lr"],
                group["b1"],
                group["b2"],
                group["wd"],
                group["eps"],
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                # Gradient of the loss at the current time step.
                grad = p.grad.data
                # Update the first moment estimate.
                m = b1 * m + (1 - b1) * grad
                # Update the second moment estimate.
                v = b2 * v + (1 - b2) * (grad**2)
                # Adjust m and v for iteration t to compensate for initialy empty,
                # i.e. zero estimate "".
                m_hat = m / (1 - b1**t)
                v_hat = v / (1 - b2**t)
                # Update the parameters.
                p.data -= lr * m_hat / (torch.sqrt(v_hat) + eps)
                # Apply weight decay - pull the parameters towards 0.
                p.data -= lr * wd * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def learning_rate_schedule(
    t: int, a_min: float, a_max: float, T_w: int, T_c: int
) -> float:
    assert T_c != T_w
    if t < T_w:
        return a_max * t / T_w
    if t <= T_c:
        return a_min + 0.5 * (a_max - a_min) * (
            1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))
        )
    return a_min


def compute_gradient_stats(parameters: Iterable[torch.nn.Parameter]) -> dict:
    """Compute statistics about gradients"""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return {}
    grad_norms = [torch.norm(g).item() for g in grads]
    return {
        'grad_mean': np.mean(grad_norms),
        'grad_std': np.std(grad_norms),
        'grad_max': max(grad_norms),
        'grad_min': min(grad_norms)
    }

def compute_activation_stats(model: nn.Module) -> dict:
    """Compute statistics about layer activations and parameters"""
    stats = {}
    activation_dict = {}

    # Hook to capture activations
    def hook_fn(name):
        def hook(module, input, output):
            activation_dict[name] = output.detach()
        return hook

    # Register hooks for all modules
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm)) or 'attention' in name.lower():
            handles.append(module.register_forward_hook(hook_fn(name)))

    # Record parameter norms
    for name, param in model.named_parameters():
        if param.requires_grad:
            stats[f'{name}_norm'] = torch.norm(param).item()

    # Record activation statistics
    for name, activation in activation_dict.items():
        if isinstance(activation, torch.Tensor):
            stats[f'{name}_act_mean'] = torch.mean(activation).item()
            stats[f'{name}_act_std'] = torch.std(activation).item()
            stats[f'{name}_act_norm'] = torch.norm(activation).item()

    # Remove the hooks
    for handle in handles:
        handle.remove()

    return stats

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float | None, eps=1e-6
) -> None:
    if max_l2_norm is None:
        return
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)


def fix_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def data_loading(
    ids: np.array,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
    always_return_same_batch: bool = False,
    vocab_size: int | None = None,
    validation: bool = False,
    starts_cnt_dict: dict[int, int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deliverable: Write a function that takes a numpy array x (integer array with token IDs), a
    batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), and returns
    a pair of tensors: the sampled input sequences and the corresponding next-token targets. Both ten-
    sors should have shape (batch_size, context_length) containing token IDs, and both should be
    placed on the requested device."""
    if always_return_same_batch:
        starts = np.zeros(batch_size, dtype=int)
    else:
        starts = np.random.randint(0, len(ids) - context_length, size=batch_size)
    if validation:
        starts = np.arange(batch_size)
    if starts_cnt_dict is not None and not validation:
        for s in starts:
            starts_cnt_dict[s] += 1
    input_seqs = np.stack([ids[s : s + context_length] for s in starts])
    output_seqs = np.stack([ids[s + 1 : s + context_length + 1] for s in starts])
    assert input_seqs.shape == (batch_size, context_length)
    assert output_seqs.shape == (batch_size, context_length)
    out = (
        torch.tensor(input_seqs, device=device, dtype=torch.long),
        torch.tensor(output_seqs, device=device, dtype=torch.long),
    )
    if vocab_size is not None:
        for tensor in out:
            assert tensor.max().item() < vocab_size
            assert tensor.min().item() >= 0
    return out


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    loss: float | None = None,
) -> None:
    """should dump all the state from the
    first three parameters into the file-like object out. You can use the state_dict method of both
    the model and the optimizer to get their relevant states and use torch.save(obj, out) to dump
    obj into out (PyTorch supports either a path or a file-like object here). A typical choice is to
    have obj be a dictionary, but you can use whatever format you want as long as you can load your
    checkpoint later.
    This function expects the following parameters:
    * model: torch.nn.Module
    * optimizer: torch.optim.Optimizer
    * iteration: int
    * out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    """
    if getattr(model, "_orig_mod", None) is None:
        model_state = model.state_dict()
    else:
        model_state = model._orig_mod.state_dict()
    optim_state = optimizer.state_dict()
    data = {"model": model_state, "optimizer": optim_state, "iteration": iteration}
    if loss is not None:
        data["loss"] = loss
    torch.save(data, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """should load a checkpoint from src (path or file-
    like object), and then recover the model and optimizer states from that checkpoint. Your
    function should return the iteration number that was saved to the checkpoint. You can use
    torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
    load_state_dict method in both the model and optimizers to return them to their previous
    states.
    This function expects the following parameters:
     * src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
     * model: torch.nn.Module
     * optimizer: torch.optim.Optimizer
    """
    saved_dict = torch.load(src)
    model.load_state_dict(saved_dict["model"])
    optimizer.load_state_dict(saved_dict["optimizer"])
    return saved_dict["iteration"]


def load_previous_losses(dir: str | os.PathLike) -> tuple[np.array, np.array]:
    return (
        np.load(os.path.join(dir, "losses_train.npy")).tolist(),
        np.load(os.path.join(dir, "losses_valid.npy")).tolist(),
        np.load(os.path.join(dir, "learning_rates.npy")).tolist()
    )


def find_latest_checkpoint(out_dir):
    # List all files in the directory
    files = os.listdir(out_dir)
    # Match files with pattern "iter_{i}"
    pattern = re.compile(r"iter_(\d+)$")
    max_iter = -1
    latest_file = None
    for fname in files:
        match = pattern.match(fname)
        if match:
            i = int(match.group(1))
            if i > max_iter:
                max_iter = i
                latest_file = os.path.join(out_dir, fname)
    return latest_file, max_iter


def training_loop(
    tokenizer: Tokenizer,
    training_steps: int,
    save_ckpt_every: int,
    array_with_training_text_tokens: np.array,
    batch_size: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    out_dir: str | os.PathLike,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    val_ids: np.ndarray = None,
    val_every: int = 500,
    always_train_on_the_same_batch: bool = False,
    # Hyperparameters for the scheduler
    max_lr: float = 3e-5,
    min_lr: float = 3e-6,
    warmup_steps: int = 2000,
    gradient_clipping_max_l2_norm: float | None = None,
    adamw_lr: float = 0.001,
    adamw_betas: tuple = (0.9, 0.999),
    adamw_weight_decay: float = 0.01,
):
    
    assert len(tokenizer.special_tokens) == 1
    eos_token = tokenizer.special_token_to_index["<|endoftext|>"]

    starts_cnt_dict = Counter()

    fix_seeds()
    start_time = time.time()
    os.makedirs(out_dir, exist_ok=True)
    model = Transformer(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        rope_theta,
        device,
        dtype,
    )

    # optimizer = AdamW(model.parameters(), lr=adamw_lr, betas=adamw_betas, weight_decay=adamw_weight_decay,)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=adamw_lr,
        betas=adamw_betas,
        weight_decay=adamw_weight_decay,
    )

    # Try to resume from latest checkpoint
    latest_ckpt, last_iter = find_latest_checkpoint(out_dir)
    if latest_ckpt is not None:
        print(f"Resuming from checkpoint {latest_ckpt} at iteration {last_iter}")
        load_checkpoint(latest_ckpt, model, optimizer)
        losses, val_losses, lrs = load_previous_losses(out_dir)
        assert (
            len(losses) == last_iter
        ), f"Loaded wrong number of losses ({len(losses)}) from the previous training run (last_iter={last_iter})"
    else:
        losses = []
        val_losses = []
        last_iter = 0
        lrs = []

    # JIT-compilation
    model = torch.compile(model, backend="aot_eager")

    # Initialize stats tracking
    gradient_stats_history = []
    activation_stats_history = []
    
    try:
        for i in range(last_iter + 1, training_steps + 1):
            model.train()
            optimizer.zero_grad()
            # TODO: make max_steps in lr schedule variable
            lr = learning_rate_schedule(i, min_lr, max_lr, warmup_steps, training_steps)
            lrs.append(lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y = data_loading(
                array_with_training_text_tokens,
                batch_size,
                context_length,
                device=device,
                always_return_same_batch=always_train_on_the_same_batch,
                vocab_size=vocab_size,
                starts_cnt_dict=starts_cnt_dict,
            )
            pred = model(x)
            assert (
                pred.shape[-1] == vocab_size
            ), "Expected logits per each token in vocab."
            loss = cross_entropy(pred, y)
            # Check for invalid loss
            if not torch.isfinite(loss):
                raise ValueError("Loss is NaN or Inf")
            losses.append(loss.cpu().item())
            loss.backward()

            # Collect gradient and activation statistics
            grad_stats = compute_gradient_stats(model.parameters())
            activation_stats = compute_activation_stats(model)
            gradient_stats_history.append(grad_stats)
            activation_stats_history.append(activation_stats)

            gradient_clipping(model.parameters(), gradient_clipping_max_l2_norm)
            optimizer.step()

            # Log statistics periodically
            if i % val_every == 0:
                print("\nGradient Statistics:")
                for k, v in grad_stats.items():
                    print(f"{k}: {v:.6f}")
                print("\nActivation Statistics:")
                for k, v in activation_stats.items():
                    print(f"{k}: {v:.6f}")
                print()

            # --- Validation ---
            if val_ids is not None and (i % val_every == 0 or i == training_steps):
                model.eval()
                with torch.no_grad():
                    x_val, y_val = data_loading(
                        val_ids,
                        batch_size,
                        context_length,
                        device=device,
                        vocab_size=vocab_size,
                        validation=True,
                    )
                    val_logits = model(x_val)
                    val_loss = cross_entropy(val_logits, y_val).cpu().item()
                    val_losses.append(val_loss)
                print(
                    f"[Step {i}] train_loss: {losses[-1]:.4f} | val_loss: {val_loss:.4f} | lr: {lr:.6f}"
                )
                start_cnts = starts_cnt_dict.most_common()
                total_visits = sum(starts_cnt_dict.values())
                unique_starts = len(starts_cnt_dict)
                print(f"Most common start: {start_cnts[0]}, Least common start: {start_cnts[-1]}, Avg cnt per start: {total_visits / unique_starts}, Unique starts: {unique_starts}")
                # decode a sentence
                prompt_tokens = x_val[0][:100]
                decoded_tokens = decode(prompt_tokens, model, max_decoding_steps=50, eos_token_id=eos_token)
                txt_input = tokenizer.decode(prompt_tokens.cpu().numpy())
                txt_genrated = tokenizer.decode(decoded_tokens.cpu().numpy())
                print(f"PROMPT\n{txt_input}")
                print(f"GENERATION\n{txt_genrated}")

            def save_checkpoint_and_stats():
                fp = os.path.join(out_dir, f"iter_{i}")
                save_checkpoint(model, optimizer, i, fp, loss=loss)
                np.save(os.path.join(out_dir, "losses_train"), losses)
                np.save(os.path.join(out_dir, "losses_valid"), val_losses)
                np.save(os.path.join(out_dir, "learning_rates"), lrs)
                np.save(os.path.join(out_dir, "gradient_stats.npy"), gradient_stats_history)
                np.save(os.path.join(out_dir, "activation_stats.npy"), activation_stats_history)
            
            # --- Save a checkpoint ---
            if i % save_ckpt_every == 0 or i == training_steps:
                save_checkpoint_and_stats()
                
            # --- Stop when started overfitting ---
            if val_ids is not None and (i % val_every == 0 or i == training_steps) and abs(val_losses[-1] - losses[-1]) > 1:
                # Save final state before stopping
                save_checkpoint_and_stats()
                raise KeyboardInterrupt("Started overfitting")
            
    except KeyboardInterrupt as e:
        print(e)

    elapsed = time.time() - start_time
    print(f"Training finished/ stopped in {elapsed/60:.2f} minutes.")
    plot_losses_and_learning_rates(losses, val_losses, lrs, val_every, out_dir)
    plot_training_statistics(gradient_stats_history, activation_stats_history, training_steps, out_dir)
    return losses, val_losses
