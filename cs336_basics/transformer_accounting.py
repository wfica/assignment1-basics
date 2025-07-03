from dataclasses import dataclass, field
import torch
from torchinfo import summary
from cs336_basics.transformer_impl import Transformer
import re

def print_str(s):
    def repl(match):
        num = int(match.group())
        if num >= 1_000_000_000_000:
            return f"{num // 1_000_000_000_000:_}T"
        elif num >= 1_000_000_000:
            return f"{num // 1_000_000_000:_}G"
        elif num >= 1_000_000:
            return f"{num // 1_000_000:_}M"
        elif num >= 1_000:
            return f"{num // 1_000:_}K"
        else:
            return f"{num:_}"
    print(re.sub(r'\d+', repl, s))

@dataclass
class TransformerSetup:
    vocab_size: int = 50_257
    context_length: int = 1_024
    num_layers: int = 48
    d_model: int = 1_600
    num_heads: int = 25
    d_ff: int = 6_400
    dtype: torch.dtype = torch.float32
    d_k: int = field(init=False)  # K and Q dimention
    d_v: int = field(init=False)  # V dimention

    def __post_init__(self):
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_model // self.num_heads

def get_num_trainable_params_in_transformer_block(ts: TransformerSetup) -> int:
    rms_norms = (
        2 * ts.d_model
    )  # two rms norms, each with d_model 'gain' (weight) params
    attention = sum(
        [
            2 * ts.d_model * ts.num_heads * ts.d_k,  # Q and K
            ts.d_model * ts.num_heads * ts.d_v,  # V
            ts.num_heads * ts.d_v * ts.d_model,  # O
        ]
    )
    ff = 3 * ts.d_model * ts.d_ff  # swiglu
    return rms_norms + attention + ff


def get_num_trainable_params(ts: TransformerSetup) -> int:
    embedding_layer = ts.vocab_size * ts.d_model
    # one rope module is shared between all transformer blocks
    rope_module = 2 * ts.d_k * ts.context_length  # precomputed Cosines and Sines
    transformer_blocks = ts.num_layers * get_num_trainable_params_in_transformer_block(
        ts
    )
    final_rms_norm = ts.d_model
    final_projection = ts.d_model * ts.vocab_size
    return (
        embedding_layer
        + rope_module
        + transformer_blocks
        + final_rms_norm
        + final_projection
    )

def print_return(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if type(result) == tuple:
            print_result = result[0]
        else:
            print_result = result
        print_str(f"{func.__name__} returned: {print_result} FLOPS")
        return result
    return wrapper

@print_return
def get_flops_rms_norm(ts: TransformerSetup) -> int:
    "Assumes batch_size = 1 and seq_len = context_len"
    scaling_factor = 2 * ts.context_length * ts.d_model  + ts.context_length + ts.context_length # square, sum, div, sqrt
    scaled_activations = ts.context_length * ts.d_model
    weighted_scaled_activations = ts.context_length * ts.d_model
    # Roughly 4 * seq_len * d_model
    return scaling_factor + scaled_activations + weighted_scaled_activations


@print_return
def get_flops_attention(ts: TransformerSetup) -> int:
    "Assumes batch_size = 1 and seq_len = context_len"
    Q_and_K = 2 * ts.num_heads * 2 *  ts.context_length * ts.d_model * ts.d_k 
    V = ts.num_heads * 2 *  ts.context_length * ts.d_model * ts.d_v
    Q_and_K_rotations = 2 * ts.num_heads * ts.context_length * ts.d_k
    mask = ts.context_length * ts.context_length
    weights = ts.num_heads * 2 * ts.context_length * ts.d_k * ts.context_length
    soft_max = ts.num_heads * 3 * ts.context_length * ts.context_length
    values = ts.num_heads * 2 * ts.context_length * ts.context_length * ts.d_v
    output = 2 * ts.context_length * (ts.num_heads * ts.d_v) * ts.d_model
    # Has basically all in all exponents and combinations
    return Q_and_K + V + Q_and_K_rotations + mask + weights + soft_max + values + output

@print_return
def get_flops_ff(ts: TransformerSetup) -> int:
    "Assumes batch_size = 1 and seq_len = context_len"
    return 3 * 2 * ts.context_length * ts.d_model * ts.d_ff  

@print_return
def get_num_flops_per_transformer_block(ts: TransformerSetup) -> int:
    "Assumes batch_size = 1 and seq_len = context_len"
    rms_norms = 2 * get_flops_rms_norm(ts)
    attention = get_flops_attention(ts)
    ff = get_flops_ff(ts)
    additions = 2 * ts.context_length * ts.d_model
    return  rms_norms + attention + ff + additions

@print_return
def get_flops_final_projection(ts: TransformerSetup) -> int:
    return 2 * ts.context_length * ts.d_model * ts.vocab_size

@print_return
def get_num_flops(ts: TransformerSetup) -> int:
    "Assumes batch_size = 1 and seq_len = context_len"
    embedding_lookup = ts.context_length
    flops_per_block = get_num_flops_per_transformer_block(ts)
    blocks = ts.num_layers * flops_per_block
    final_rms_norm = get_flops_rms_norm(ts)
    final_projection = get_flops_final_projection(ts)
    return (embedding_lookup + blocks + final_rms_norm + final_projection, flops_per_block)

def transformer_accounting_a():
    gpt_2_xl = TransformerSetup()

    num_params = get_num_trainable_params(gpt_2_xl)
    bytes_per_param = gpt_2_xl.dtype.itemsize

    print_str(
        f"GPT 2 XL has\n - {num_params} trainable params\n - occupying {(num_params * bytes_per_param)} bytes of memory assuming each is a {gpt_2_xl.dtype}."
    )
    print("--------------------")


def _get_transformer_flops(ts: TransformerSetup, name: str):
    flops_total, flops_per_transformer_block = get_num_flops(ts)
    print_str(
        f"{name} forawrd pass uses:\n - {flops_total} FLOPS in total,\n - {flops_per_transformer_block}  FLOPS per transformer block"
    )
    print("--------------------")

def transformer_flops_b():
    _get_transformer_flops(TransformerSetup(), "gpt-2-xl")

def transformer_flops_c():
    print("The most expenisve (in FLOPS) single layer of gpt-2-xl is the final projection layer.")
    print("--------------------")

def transformer_flops_d():
    _get_transformer_flops(TransformerSetup(num_layers=12, d_model=768, num_heads=12), "gpt-2-small")
    _get_transformer_flops(TransformerSetup(num_layers=24, d_model=1024, num_heads=16), "gpt-2-medium")
    _get_transformer_flops(TransformerSetup(num_layers=36, d_model=1280, num_heads=20), "gpt-2-large")


def transformer_accounting_a_torchinfo():
    gpt_2_xl = TransformerSetup()
    model = Transformer(
        gpt_2_xl.vocab_size,
        gpt_2_xl.context_length,
        gpt_2_xl.num_layers,
        gpt_2_xl.d_model,
        gpt_2_xl.num_heads,
        gpt_2_xl.d_ff,
        0.123,
    )
    print(summary(model, input_size=(1, gpt_2_xl.context_length)))
    print("--------------------")


if __name__ == "__main__":
    transformer_accounting_a()
    transformer_flops_b()
    transformer_flops_c()
    transformer_flops_d()

