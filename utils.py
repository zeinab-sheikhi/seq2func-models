import math
import numpy as np 
import torch
from typing import List


def get_out_channels(start: int, end: int, n_blocks: int) -> List[int]:
    return [int(round(x)) for x in np.linspace(start, end, n_blocks)]


def basis_function_exponential(r: torch.Tensor, l: int , steps: int):
    # r: (2*l -1,)
    half_lives = torch.logspace(math.log(3), math.log(l), steps=steps, base=math.e)  # (steps,)
    abs_r = r.abs().unsqueeze(1)
    sign = torch.sign(r).unsqueeze(1)
    value = torch.exp(-math.log(2) * abs_r / half_lives)
    return torch.concat([value, sign * value], dim=1)  # (2*l-1, 2*steps)


def basis_function_central_mask():
    pass


def basis_function_gamma():
    pass
