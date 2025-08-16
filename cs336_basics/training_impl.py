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
    nll = -torch.gather(logits_scaled, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) + torch.log(
        torch.sum(e_logits_scaled, dim=-1)
    )
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
    if t < T_w:
        return a_max * t / T_w
    if t <= T_c:
        return a_min + 0.5 * (a_max - a_min) * (
            1 + math.cos(math.pi * (t - T_w) / (T_w - T_c))
        )
    return a_min


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
    vocab_size: int | None = None
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
    model_state = model.state_dict()
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


def find_latest_checkpoint(out_dir):
    return None, None
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
):
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

    # optimizer = AdamW(model.parameters())
    # for name, p in model.named_parameters():
    #     print(name, p.size())
    optimizer = torch.optim.AdamW(model.parameters())

    losses = []
    val_losses = []

    # Try to resume from latest checkpoint
    latest_ckpt, start_iter = find_latest_checkpoint(out_dir)
    if latest_ckpt is not None:
        print(f"Resuming from checkpoint {latest_ckpt} at iteration {start_iter}")
        load_checkpoint(latest_ckpt, model, optimizer)
    else:
        start_iter = 0

    for i in range(1, training_steps + 1):
        model.train()
        optimizer.zero_grad()
        # TODO: make max_steps in lr schedule variable
        lr = learning_rate_schedule(i, min_lr, max_lr, warmup_steps, training_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        x, y = data_loading(
            array_with_training_text_tokens,
            batch_size,
            context_length,
            device=device,
            always_return_same_batch=always_train_on_the_same_batch,
            vocab_size=vocab_size
        )
        pred = model(x)
        assert pred.shape[-1] == vocab_size, "Expected logits per each token in vocab."
        loss = cross_entropy(pred, y)
        # loss = F.cross_entropy(pred.view(-1, pred.size(-1)), y.view(-1))
        losses.append(loss.cpu().item())
        loss.backward()

        gradient_clipping(model.parameters(), gradient_clipping_max_l2_norm)
        # print([p.abs().mean().item() for p in model.parameters()])
        optimizer.step()
        # print([p.abs().mean().item() for p in model.parameters()])

        if i % save_ckpt_every == 0 or i == training_steps:
            fp = os.path.join(out_dir, f"iter_{i}")
            save_checkpoint(model, optimizer, i, fp, loss=loss)
        # --- Validation ---
        if val_ids is not None and (i % val_every == 0 or i == training_steps):
            model.eval()
            with torch.no_grad():
                x_val, y_val = data_loading(
                    val_ids, batch_size, context_length, device=device, vocab_size=vocab_size
                )
                val_logits = model(x_val)
                val_loss = cross_entropy(val_logits, y_val).cpu().item()
                # val_loss = F.cross_entropy(val_logits, y_val).item()
                # val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), y_val.view(-1)).item()
                val_losses.append(val_loss)
            print(
                f"[Step {i}] Training loss: {losses[-1]:.4f} | Validation loss: {val_loss:.4f}"
            )

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed/60:.2f} minutes.")
    return losses, val_losses

