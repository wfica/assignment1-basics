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


class RotaryPositionalEmbedding(nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """Construct the RoPE module and create buffers if needed.
        * theta: float Θ value for the RoPE
        * d_k: int dimension of query and key vectors
        * max_seq_len: int Maximum sequence length that will be inputted
        * device: torch.device | None = None Device to store the buffer on"""
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        positions = torch.arange(
            0, max_seq_len, dtype=torch.float32, device=device
        )  # (max_seq_len)
        exponents = theta ** (
            -torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k
        )
        exponents = torch.stack([exponents, exponents], dim=-1).flatten()  # (d_k)
        Thetas = einsum(positions, exponents, "l, d -> l d")
        self.register_buffer(
            "Cosines", torch.cos(Thetas), persistent=False
        )  # (max_seq_len, d_k)
        self.register_buffer(
            "Sines", torch.sin(Thetas), persistent=False
        )  # (max_seq_len, d_k)

    def _flip_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Given vector (x_1, x_2, x_3, ..., x_n) returns (-x_2, x_1, -x_4, x_3, ..., -x_n, x_{n-1}) Supports arbitrary batch dimentions before the final dimention."""
        x_even = torch.gather(
            x, -1, torch.arange(0, x.shape[-1], 2).expand(x.shape[:-1] + (-1,))
        )
        x_odd = torch.gather(
            x, -1, torch.arange(1, x.shape[-1], 2).expand(x.shape[:-1] + (-1,))
        )
        x_flipped = torch.stack([-x_odd, x_even], dim=-1).flatten(start_dim=-2)
        assert x_flipped.shape == x.shape
        return x_flipped

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions.
        You should assume that the token positions are a tensor of shape (..., seq_len)
        specifying the token positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed)
        cos and sin tensors along the sequence dimension.
         * x: input tensor of shape (..., seq_len, d_k)
         * token_positions: a tensor of shape (..., seq_len)"""
        x_flipped = self._flip_tensor(x)
        rotated_x = (
            self.Cosines[token_positions] * x + self.Sines[token_positions] * x_flipped
        )
        return rotated_x


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_scaled = x - torch.max(x, dim=dim, keepdim=True)[0]
    e_scaled_x = torch.exp(x_scaled)
    return e_scaled_x / torch.sum(e_scaled_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Attention(Q, K, V) = softmax(Q.T @ K / √d_k) @ V
    * Q.shape = [batch_size, ..., seq_len, d_k]
    * K.shape = [batch_size, ..., seq_len, d_k]
    * V.shape = [batch_size, ..., seq_len, d_v]
    * mask.shape = [seq_len, seq_len]
    Returns a rensor of shape [batch_size, ..., seq_len, d_v]"""
    weights = einsum(q, k, "b ... n1 k, b ... n2 k -> b ... n1 n2") / (
        q.shape[-1] ** 0.5
    )
    if mask is not None:
        mask = torch.where(
            mask,
            torch.zeros_like(mask, dtype=weights.dtype),
            torch.full_like(mask, float("-inf"), dtype=weights.dtype),
        )
        weights += mask
    normed_weights = softmax(weights, dim=len(weights.shape) - 1)
    out = einsum(normed_weights, v, "b ... n1 n2, b ... n2 d_v -> b ... n1 d_v")
    return out


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_module: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.rope = rope_module
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.Wq = Linear(
            self.d_model, self.num_heads * self.d_k, device=device, dtype=dtype
        )
        self.Wk = Linear(
            self.d_model, self.num_heads * self.d_k, device=device, dtype=dtype
        )
        self.Wv = Linear(
            self.d_model, self.num_heads * self.d_v, device=device, dtype=dtype
        )
        self.Wo = Linear(
            self.num_heads * self.d_v, self.d_model, device=device, dtype=dtype
        )

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
         * x.shape = [..., seq_len, d_model]
         * token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens
        Returns a vector of the same shape"""
        seq_len = x.shape[-2]
        assert self.rope is None or token_positions is not None
        token_positions = (
            token_positions if token_positions is not None else torch.arange(seq_len)
        )
        Q = rearrange(
            self.Wq(x),
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        Q_rotated = self.rope(Q, token_positions) if self.rope is not None else Q
        K = rearrange(
            self.Wk(x),
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        K_rotated = self.rope(K, token_positions) if self.rope is not None else K
        V = rearrange(
            self.Wv(x),
            "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",
            num_heads=self.num_heads,
        )
        # casual attention mask
        mask = (1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1)).to(bool)
        atten_output = scaled_dot_product_attention(Q_rotated, K_rotated, V, mask)
        atten_output_concatenated = rearrange(
            atten_output,
            "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)",
            num_heads=self.num_heads,
        )
        return self.Wo(atten_output_concatenated)
