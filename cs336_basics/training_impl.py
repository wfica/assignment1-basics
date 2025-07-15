import torch
from torch import nn

import numpy as np
from einops import rearrange, einsum


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
