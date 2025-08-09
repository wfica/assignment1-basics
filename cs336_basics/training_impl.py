import torch
from torch import nn
from collections.abc import Callable, Iterable
from typing import Optional
import math
import numpy as np
from einops import rearrange, einsum
import matplotlib.pyplot as plt


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> float:
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
    print(logits.shape, targets.shape)
    logits_scaled = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    e_logits_scaled = torch.exp(logits_scaled)
    nll = -torch.gather(logits_scaled, dim=-1, index=targets.unsqueeze(-1)) + torch.log(
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


def toy_training_example(optimizer, training_steps=10, **optimizer_defaults):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = optimizer([weights], **optimizer_defaults)
    losses = []
    for t in range(training_steps):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.cpu().item())
        loss.backward()
        opt.step()
    return losses


def test_toy_training():
    lrs = [1e1, 1e2, 1e3]
    losses = [toy_training_example(SGD, lr=lr) for lr in lrs]
    for lr, loss in zip(lrs, losses):
        print(lr, [int(l) for l in loss])


# 10.0 [23, 15, 11, 8, 7, 5, 5, 4, 3, 3] did not coverge to 0 in 10 steps
# 100.0 [21, 21, 3, 0, 0, 0, 0, 0, 0, 0] coverged to 0 in 4 steps
# 1000.0 [33, 11914, 2057819, 228910400, 18541742080, 1170196398080, 60074051502080, 2584642867691520, 95264410439778304, 3059046407391412224] diverged


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


def test_toy_adamw_training():
    losses = toy_training_example(AdamW, training_steps=130)
    print([int(l) for l in losses])


def adamwAccounting_a():
    """
    How much peak memory does running AdamW require? Decompose your answer based on the
    memory usage of the parameters, activations, gradients, and optimizer state. Express your answer
    in terms of the batch_size and the model hyperparameters (vocab_size, context_length,
    num_layers, d_model, num_heads). Assume d_ff = 4 x d_model, d_k = d_v = d_model / num_heads.
    For simplicity, when calculating memory usage of activations, consider only the following components:
      * Transformer block
        * RMSNorm(s)
        * Multi-head self-attention sublayer: QKV projections, Q^{T} @ K matrix multiply, softmax, weighted sum of values, output projection.
        * Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
      * final RMSNorm
      * output embedding
      * cross-entropy on logits
    Deliverable: An algebraic expression for each of parameters, activations, gradients, and optimizer state, as well as the total.

    PARAMS:
    shared ROPE: 2 * seq_len * d_k = 2 * seq_len * d_model / num_heads
    Embeddings: vocab_size * embedding_dim = vocab_size * d_model
    num_layers x Transformer block:
        * 2 x RMS: 2 * d_model (gain)
        * Wq, Wk, Wv, Wo: 4 * d_model^2
        * FF: 3 * d_model * d_ff = 12 * d_model^2
    final RMS: d_model
    final FF: d_model * vocab_size

    total PARAMS:
      + 2 * seq_len * d_model / num_heads
      + num_layers * (2 * d_model + 16 * d_model^2)
      + d_model
      + 2 * vocab_size * d_model

    ACTIVATIONS:
    Embeddings: bs * sl * dm
    num_layers x Transformer block:
        * 2 x RMS: 2 * (bs * sl + 2 * bs * sl * dm)
        * QKV projections: 3 * bs * sl * dm
        * QK rotations: 2 * bs * sl * dm
        * Q^{T} @ K: bs * num_heads * sl^2
        * soft_maxed weights: bs * num_heads * sl^2
        * attention: bs * sl * dm
        * out projection: bs * sl * dm
        * W1, silu, W2: 2 * bs * sl * d_ff + bs * sl * dm = 9*bs*sl*dm
    final RMS: bs * sl + 2 * bs * sl * dm
    final FF: bs * sl * vocab_size

    total ACTIVATIONS (some smaller terms ommited):
     + 3 * bs * sl * dm  + bs * sl * vocab_size
     + num_layers * (
         + 16 * bs * sl * dm
         + 2 * bs * num_heads * sl^2
     )

    GRADIENTS
      * same as PARAMS (needs gradient with respect to each param) - ROPE params
    does not need ACTIVATIONS (needs gradients with respect to activations so that can backpropagate but does not need to store them, can be temp computed and discarded as soon as are used to compute gradients with respect to the relevant params)

    ADAMW OPTIMIZER STATE:
      * 2 x PARAMS for storing estimates of the first and second moments

    in total: 4 * PARAMS + ACTIVATIONS, that is approximately

    + 4 * ( 2 * sl * dm / nh + 16 * nl * dm^2 + 2 * vs * dm)
    + bs * (3 * sl * dm + sl * vs + nl * (16 * sl * dm + 2 * nh * sl^2))
    """

    print(
        """Transformer model needs 4 * PARAMS + ACTIVATIONS peak memory, that approximately is 
    + 4 * ( 2 * sl * dm / nh + 16 * nl * dm^2 + 2 * vs * dm) 
    + bs * (3 * sl * dm + sl * vs + nl * (16 * sl * dm + 2 * nh * sl^2))"""
    )
    print("This roughly agrees with https://erees.dev/transformer-memory/\n")


def adamwAccounting_b():
    """
        Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on
    the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?
    Deliverable: An expression that looks like a·batch_size + b for numerical values a, b, and a
    number representing the maximum batch size.
    """
    # Defaults for GPT-2-XL
    vs: int = 50_257
    sl: int = 1_024
    nl: int = 48
    dm: int = 1_600
    nh: int = 25
    d_ff: int = 6_400
    assert d_ff == 4 * dm

    max_mem = 80_000_000_000

    num_static_mem = 4 * (2 * sl * dm / nh + 16 * nl * dm**2 + 2 * vs * dm)
    num_activations = 3 * sl * dm + sl * vs + nl * (16 * sl * dm + 2 * nh * sl**2)

    # Assume all params and activations are stored on float32, i.e. 4B
    params = 4 * num_static_mem
    activations = 4 * num_activations

    print(
        f"Estimate for GPT-2-XL static_mem and activations:\nparams = {params:.2e} bytes \nactivations = {activations:.2e} bytes"
    )

    max_bs = (max_mem - params) // activations

    print(f"Max batch_size = {max_bs} for GPT-2-XL running on 80GB memory.\n")


def adamwAccounting_c():
    """How many FLOPs does running one step of AdamW take?
    Deliverable: An algebraic expression, with a brief justification.

        We assume the approximation that
          * Forward pass: 2 (# data points) (# parameters) FLOPs
          * Backward pass: 4 (# data points) (# parameters) FLOPs
        At this point we have the activations and the gradients.
        We need to go over all the params and update their 2 moments estimates and also update themselves. That is k * (# parameters) FLOPs for some constant k.
    """
    print(
        """How many FLOPs does running one step of AdamW take?
In total that is 
    6 * (bs * sl) * params + k * params
for 
    params = ( 2 * sl * dm / nh + 16 * nl * dm^2 + 2 * vs * dm).
That is roughly 6 (#tokens) (#params)
Which agrees with https://www.adamcasson.com/posts/transformer-flops#user-content-fn-flops_calc.
    """
    )


def adamwAccounting_d():
    """Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second)
    relative to the hardware’s theoretical peak FLOP throughput [Chowdhery et al., 2022]. An
    NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming
    you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a
    batch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022],
    assume that the backward pass has twice the FLOPs of the forward pass.
    Deliverable: The number of days training would take, with a brief justification.
    """
    data_points_in_batch = 1024 * 1024  # (batch * seq_len)
    total_flops_per_step = 6 * 1.5e9 * data_points_in_batch
    total_flops = total_flops_per_step * 4e5
    flop_per_day = 19.5e12 * 0.5 * 3600 * 24
    num_days = total_flops / flop_per_day
    print(f"It would take {int(num_days)} days.")


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


# run with uv `run -m cs336_basics.training_impl`
if __name__ == "__main__":
    print("SDG test:")
    # SDG test:
    # 10.0 [21, 13, 10, 7, 6, 5, 4, 3, 3, 2]
    # 100.0 [23, 23, 4, 0, 0, 0, 0, 0, 0, 0]
    # 1000.0 [29, 10492, 1812259, 201594560, 16329158656, 1030556745728, 52905419735040, 2276217239633920, 83896516770529280, 2694010471115128832]
    test_toy_training()
    print("AdamW test:")
    # AdamW test:
    # [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23]

    test_toy_adamw_training()
    print("AdamW accounting:")
    adamwAccounting_a()
    # AdamW accounting:
    # Transformer model needs 4 * PARAMS + ACTIVATIONS peak memory, that approximately is
    #     + 4 * ( 2 * sl * dm / nh + 16 * nl * dm^2 + 2 * vs * dm)
    #     + bs * (3 * sl * dm + sl * vs + nl * (16 * sl * dm + 2 * nh * sl^2))
    # This roughly agrees with https://erees.dev/transformer-memory/
    adamwAccounting_b()
    # Estimate for GPT-2-XL static_mem and activations:
    # params = 3.40e+10 bytes
    # activations = 1.53e+10 bytes
    # Max batch_size = 2.0 for GPT-2-XL running on 80GB memory.
    adamwAccounting_c()
    # How many FLOPs does running one step of AdamW take?
    # In total that is
    #     6 * (bs * sl) * params + k * params
    # for
    #     params = ( 2 * sl * dm / nh + 16 * nl * dm^2 + 2 * vs * dm).
    # That is roughly 6 (#tokens) (#params)
    # Which agrees with https://www.adamcasson.com/posts/transformer-flops#user-content-fn-flops_calc.
    adamwAccounting_d()
    # It would take 4481 days.
