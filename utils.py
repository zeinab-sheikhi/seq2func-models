import math
import numpy as np 
import torch
from typing import List


def get_out_channels(start: int, end: int, n_blocks: int) -> List[int]:
    return [int(round(x)) for x in np.linspace(start, end, n_blocks)]


def positional_features_exponential(
    positions: torch.Tensor,
    feat_size: int,
    seq_len: int | None = None,
    min_half_life: float | None = 3.0, 
):
    """ Create exponentially deacying positional weights.

    Args:
        positions: Position Tensor (r in the paper)
        feat_size: number of basis functions to use
        seq_len: Sequence length
        min_half_life: Smallest exponential half life in the grid of half lives.
    
    Returns:
        A tensor with shape [2 * seq_len - 1, 2 * feat_size]
    """
    if seq_len is None:
        seq_len = int(positions.abs().max().item()) + 1

    assert positions.shape == torch.Size([2 * seq_len - 1]), \
        f"positions tensor must have shape (2 * seq_len - 1), got {positions.shape}"
    
    half_lives = torch.logspace(min_half_life, math.log2(seq_len), 
                                steps=feat_size, base=2)  # (feat_size,)
    abs_r = positions.abs().unsqueeze(1)
    sign = torch.sign(positions).unsqueeze(1)
    value = torch.exp(-math.log(2) * abs_r / half_lives)
    return torch.concat([value, sign * value], dim=1)


def positional_features_central_mask(positions: torch.Tensor, feat_size: int):
    """Positional features using a central mask (allow only central features)."""
    
    i = torch.arange(0, feat_size + 1, dtype=torch.float32)
    center_widths = 2 ** i - 1
    abs_r = positions.abs().unsqueeze(1)
    sign = torch.sign(positions).unsqueeze(1)
    value = (center_widths > abs_r).float()
    return torch.cat([value, value * sign], dim=1)


def positional_features_gamma(positions: torch.Tensor, feat_size: int, seq_len: int | None = None, stddev=None, start_mean=None):
    """Poitional features computed using Gamma distribution."""
    
    if seq_len is None:
        seq_len = int(positions.abs().max().item()) + 1
    
    assert positions.shape == torch.Size([2 * seq_len - 1]), \
        f"positions must have shape (2*seq_len-1,), got {positions.shape}"
    
    if stddev is None:
        stddev = seq_len / (2 * feat_size)
    if start_mean is None:
        start_mean = seq_len / feat_size
    
    mu = torch.linspace(start_mean, seq_len, feat_size)

    alpha = (mu / stddev) ** 2
    beta = mu / (stddev ** 2)

    dist = torch.distributions.Gamma(concentration=alpha, rate=beta)
    
    abs_r = positions.abs().unsqueeze(1).clamp(1e-8)
    sign = torch.sign(positions).unsqueeze(1)

    value = torch.exp(dist.log_prob(abs_r))
    value = value + 1e-8
    value = value / value.max(dim=0, keepdim=True).values
    return torch.cat([value, value * sign], dim=1)


