import torch
from cs336_basics.transformer_impl import Transformer, softmax


def decode_step(
    step: int,
    tokens: torch.Tensor,
    transformer: Transformer,
    temperature: float,
    nucleus_threshlod: float,
) -> int:
    """One decode step"""
    assert nucleus_threshlod >= 0 and nucleus_threshlod <= 1
    logits = transformer(tokens[None, :])[0, step-1] / temperature
    probabilities = softmax(logits, -1)
    sorted_vals, sorted_indicies = torch.sort(probabilities, descending=True)
    cum_sum = torch.cumsum(sorted_vals, dim=0)
    prefix_len = (cum_sum > nucleus_threshlod).nonzero(as_tuple=True)[0]
    if len(prefix_len) > 0:
        smallest_prefix = prefix_len[0].item() + 1  # +1 because index is zero-based
    else:
        raise ValueError("Should never happen as long as nucleus_threshlod in [0, 1]")

    prefix_indices = sorted_indicies[:smallest_prefix]
    prefix_values = sorted_vals[:smallest_prefix]

    sampled_idx = torch.multinomial(prefix_values, num_samples=1)
    return prefix_indices[sampled_idx].item()


def decode(
    input_tokens: torch.Tensor,
    transformer: Transformer,
    max_decoding_steps: int,
    eos_token_id: int,
    temperature: float = 1.0,
    nucleus_threshlod: float = 0.5,
) -> torch.Tensor:
    """Given
     * input_tokens: torch.Tensor[torch.int32]
     * transformer: Transformer
    returns:
     * generated tokens during decoding
    """
    transformer.eval()
    model_device = next(transformer.parameters()).device
    if input_tokens.device != model_device:
        input_tokens = input_tokens.to(model_device)
    tokens = torch.cat(
        [input_tokens, torch.zeros(max_decoding_steps, device=model_device, dtype=torch.int32)]
    )
    decoding_length = 0
    for i in range(len(input_tokens), len(input_tokens) + max_decoding_steps):
        new_token = decode_step(i, tokens, transformer, temperature, nucleus_threshlod)
        tokens[i] = new_token
        decoding_length += 1
        if tokens[i] == eos_token_id:
            break
    return tokens[len(input_tokens) : len(input_tokens) + decoding_length]
