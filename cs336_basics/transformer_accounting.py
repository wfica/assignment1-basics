from dataclasses import dataclass, field
import torch
from torchinfo import summary
from cs336_basics.transformer_impl import Transformer
import re
import pandas as pd

def humanize_ints_in_string(s):
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
    return re.sub(r"\d+", repl, str(s))

def humanize_with_percent(val, total):
    val_int = int(re.sub(r"[^\d]", "", str(val)))  # Remove non-digits, get int
    percent = 100 * val_int / total if total else 0
    humanized = humanize_ints_in_string(str(val_int))
    return f"{humanized} ({percent:.2f}%)"

def print_str(s):
    print(humanize_ints_in_string(s))



@dataclass
class TransformerSetup:
    # Defaults to GPT-2-XL
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
    scaling_factor = (
        2 * ts.context_length * ts.d_model + ts.context_length + ts.context_length
    )  # square, sum, div, sqrt
    scaled_activations = ts.context_length * ts.d_model
    weighted_scaled_activations = ts.context_length * ts.d_model
    # Roughly 4 * seq_len * d_model
    return {'rms_norm': scaling_factor + scaled_activations + weighted_scaled_activations}


@print_return
def get_flops_attention(ts: TransformerSetup) -> dict[str, int]:
    "Assumes batch_size = 1 and seq_len = context_len"
    Q_and_K = 2 * ts.num_heads * 2 * ts.context_length * ts.d_model * ts.d_k
    V = ts.num_heads * 2 * ts.context_length * ts.d_model * ts.d_v
    Q_and_K_rotations = 2 * ts.num_heads * ts.context_length * ts.d_k
    mask = ts.context_length * ts.context_length
    weights = ts.num_heads * 2 * ts.context_length * ts.d_k * ts.context_length
    soft_max = ts.num_heads * 3 * ts.context_length * ts.context_length
    values = ts.num_heads * 2 * ts.context_length * ts.context_length * ts.d_v
    output = 2 * ts.context_length * (ts.num_heads * ts.d_v) * ts.d_model
    # Has basically all in all exponents and combinations
    return {'atten': Q_and_K + V + Q_and_K_rotations + mask + weights + soft_max + values + output}


@print_return
def get_flops_ff(ts: TransformerSetup) -> dict[str, int]:
    "Assumes batch_size = 1 and seq_len = context_len"
    return {'ff': 3 * 2 * ts.context_length * ts.d_model * ts.d_ff}


@print_return
def get_num_flops_per_transformer_block(ts: TransformerSetup) -> dict[str, int]:
    "Assumes batch_size = 1 and seq_len = context_len"
    rms_norms = get_flops_rms_norm(ts)
    attention = get_flops_attention(ts)
    ff = get_flops_ff(ts)
    additions = 2 * ts.context_length * ts.d_model
    ret = rms_norms | attention | ff
    ret['transformer_block'] = 2 * ret['rms_norm'] + attention['atten'] + ff['ff'] + additions
    return ret


@print_return
def get_flops_final_projection(ts: TransformerSetup) -> dict[str, int]:
    return {'final_projection': 2 * ts.context_length * ts.d_model * ts.vocab_size}


@print_return
def get_num_flops(ts: TransformerSetup) -> dict[str, int]:
    "Assumes batch_size = 1 and seq_len = context_len"
    embedding_lookup = ts.context_length
    flops_per_block = get_num_flops_per_transformer_block(ts)
    blocks = ts.num_layers * flops_per_block['transformer_block']
    final_rms_norm = get_flops_rms_norm(ts)
    final_projection = get_flops_final_projection(ts)
    ret = flops_per_block | final_rms_norm | final_projection
    ret['all_blocks'] = blocks
    ret['total_flops'] = embedding_lookup + blocks + final_rms_norm['rms_norm'] + final_projection['final_projection']
    return ret


def transformer_accounting_a():
    gpt_2_xl = TransformerSetup()

    num_params = get_num_trainable_params(gpt_2_xl)
    bytes_per_param = gpt_2_xl.dtype.itemsize

    print_str(
        f"GPT 2 XL has\n - {num_params} trainable params\n - occupying {(num_params * bytes_per_param)} bytes of memory assuming each is a {gpt_2_xl.dtype}."
    )
    print("--------------------")


def _get_transformer_flops(ts: TransformerSetup, name: str):
    flops_total = get_num_flops(ts)
    print_str(
        f"{name} forawrd pass uses:\n - {flops_total['total_flops']} FLOPS in total,\n - {flops_total['transformer_block']}  FLOPS per transformer block"
    )
    print("--------------------")


def transformer_flops_b():
    _get_transformer_flops(TransformerSetup(), "gpt-2-xl")


def transformer_flops_c():
    print("""In the assumed setup:
         - The most expenisve (in FLOPS) single layer of gpt-2-xl is the final projection layer.
         - Or if transformer bloks are accounted jointly then the linear FFN part is the most expensive."""
    )
    print("--------------------")


def transformer_flops_d():
    setups = [
        ("gpt-2-S", TransformerSetup(num_layers=12, d_model=768, num_heads=12)),
        ("gpt-2-M", TransformerSetup(num_layers=24, d_model=1024, num_heads=16)),
        ("gpt-2-L", TransformerSetup(num_layers=36, d_model=1280, num_heads=20)),
        ("gpt-2-XL", TransformerSetup()),
        ("gpt-2-XL-lc", TransformerSetup(context_length=16_384)),

    ]
    data = []
    for name, ts in setups:
        flops = get_num_flops(ts)
        row = {"model": name}
        row.update(flops)
        data.append(row)
    df = pd.DataFrame(data)
    for col in df.columns:
        if col != "model":
            df[col] = [
                humanize_with_percent(val, total)
                for val, total in zip(df[col], df["total_flops"])
            ]
    print(df.T)  # Transpose for easier comparison
    return df


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

# Run with `uv run -m cs336_basics.transformer_accounting`
if __name__ == "__main__":
    transformer_accounting_a()
# GPT 2 XL has
#  - 2G trainable params
#  - occupying 8G bytes of memory assuming each is a torch.float32.
    transformer_flops_b()
# In the assumed setup:
#  - The most expenisve (in FLOPS) single layer of gpt-2-xl is the final projection layer.
#  - Or if transformer bloks are accounted jointly then the linear FFN part is the most expensive.
    transformer_flops_c()
    transformer_flops_d()
# FLOPS
# model                     gpt-2-S       gpt-2-M       gpt-2-L      gpt-2-XL     gpt-2-XL-lc
# rms_norm               3M (0.00%)    4M (0.00%)    5M (0.00%)    6M (0.00%)    104M (0.00%)
# atten                  8G (1.50%)   12G (0.94%)   18G (0.72%)   27G (0.61%)      2T (1.38%)
# ff                    30G (5.61%)   40G (2.91%)   50G (1.92%)   62G (1.39%)      1T (0.67%)
# transformer_block     38G (7.11%)   53G (3.85%)   69G (2.64%)   90G (2.01%)      3T (2.05%)
# final_projection     79G (14.67%)  105G (7.62%)  131G (5.02%)  164G (3.64%)      2T (1.75%)
# all_blocks          459G (85.32%)   1T (92.38%)   2T (94.98%)   4T (96.35%)   147T (98.25%)
# total_flops        538G (100.00%)  1T (100.00%)  2T (100.00%)  4T (100.00%)  150T (100.00%)
