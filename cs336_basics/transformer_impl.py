import torch
from torch import nn
import numpy as np
from einops import rearrange, einsum


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ):
        # super(Linear, self).__init__()
        super().__init__()
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device)
        )
        std = (2 / (in_features + out_features)) ** (1 / 2)
        nn.init.trunc_normal_(self.W, 0, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            self.W, x, "out_features in_features, ... in_features -> ... out_features"
        )
