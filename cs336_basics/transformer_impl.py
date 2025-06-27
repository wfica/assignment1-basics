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


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct the RMSNorm module. This function should accept the following parameters:
        * d_model: int Hidden dimension of the model
        * eps: float = 1e-5 Epsilon value for numerical stability
        * device: torch.device | None = None Device to store the parameters on
        * dtype: torch.dtype | None = None Data type of the parameters"""
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(  # gain parameter (weight)
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        in_dtype = activations.dtype
        activations = activations.to(torch.float32)  # (b s d)
        rms = torch.sqrt(torch.mean(activations**2, dim=-1) + self.eps)  # (b s)
        scaled_activations = einsum(
            activations, 1 / rms, "... d, ... -> ... d"
        )  # (b s d)
        result = einsum(scaled_activations, self.g, "... d, d -> ... d")  # (b s d)
        return result.to(in_dtype)


class FFNSwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ):
        """Constructs a FF layer with a SwiGLU activations.
        FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x).
        Does not use biases.
        * d_model: int Hidden dimension of the model
        * d_ff: int | None = None Defaults to the nearest 64-multiple of 8 / 3 * d_model.
        * device: torch.device | None = None Device to store the parameters on
        * dtype: torch.dtype | None = None Data type of the parameters"""
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = min(64, (d_ff // 64) * 64)
            assert d_ff % 64 == 0
        self.d_ff = d_ff
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.silu(self.W1(x)) * self.W3(x))
