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
                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t+1
        return loss
    
def toy_training_example(lr, training_steps=10):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr)
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
    losses = [toy_training_example(lr) for lr in lrs]
    for lr, loss in zip(lrs, losses):
      print(lr, [int(l) for l in loss])
# 10.0 [23, 15, 11, 8, 7, 5, 5, 4, 3, 3] did not coverge to 0 in 10 steps
# 100.0 [21, 21, 3, 0, 0, 0, 0, 0, 0, 0] coverged to 0 in 4 steps
# 1000.0 [33, 11914, 2057819, 228910400, 18541742080, 1170196398080, 60074051502080, 2584642867691520, 95264410439778304, 3059046407391412224] diverged

if __name__ == "__main__":
    test_toy_training()
    