import torch
from torch import nn
import numpy as np
from einops import rearrange, einsum
from jaxtyping import Float


def print_model():
    pass


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ):
        """Construct a linear transformation module. This function should accept the following parameters:
        * in_features: int final dimension of the input
        * out_features: int final dimension of the output
        * device: torch.device | None = None Device to store the parameters on
        * dtype: torch.dtype | None = None Data type of the parameters"""
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


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct an embedding module. This function should accept the following parameters:
        * num_embeddings: int Size of the vocabulary
        * embedding_dim: int Dimension of the embedding vectors, i.e., d_model
        * device: torch.device | None = None Device to store the parameters on
        * dtype: torch.dtype | None = None Data type of the parameters"""
        super().__init__()
        self.E = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        nn.init.trunc_normal_(self.E, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs.
        * token_ids: torch.LongTensor of token IDs with shape (batch_size, sequence_length)
        """
        return self.E[token_ids]
