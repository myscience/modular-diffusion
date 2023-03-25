import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Tuple, Iterable

from itertools import tee, islice, zip_longest, chain

from einops import rearrange

def exists(var : Any):
    return var is not None

def default(var : Any, val : Any):
    return var if exists(var) else val

def extract(a : Tensor, idx : Tensor, shape : Tuple) -> Tensor:
    b, *_ = idx.shape

    out = a.gather(-1, idx)
    return out.reshape(b, *((1,) * (len(shape) - 1)))

def enlarge_as(a : Tensor, b : Tensor) -> Tensor:
    '''
        Add sufficient number of singleton dimensions
        to tensor a **to the right** so to match the
        shape of tensor b. NOTE that simple broadcasting
        works in the opposite direction.
    '''
    return rearrange(a, f'... -> ...{" 1" * (b.dim() - a.dim())}').contiguous()

def pairwise(iterable : Iterable) -> Iterable:
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def groupwise(iterable, n : int = 3, pad : int = 0, extend : bool = True, fill = 0):
    groups = tee(iterable, n)
    padded = []
    for i, v in enumerate(groups):
        v = chain([fill] * (abs(pad) - i), v) if pad < 0 else chain(v, [fill] * (abs(pad) - i))
        _ = list(islice(v, 0, max(i + pad, 0)))
        padded += [v]
    
    return zip_longest(*padded, fillvalue = fill) if extend else zip(*padded)

class BatchDrop(nn.Module):
    '''
        A custom module that randomly zeros-out an element in
        a batch independently of input shape. Input is assumed
        to have shape [B, ...] and each tensor in batch is
        zeroed-out to implement condition dropping in classifier-
        free guidance for diffusive model
    '''
    def __init__(self, p) -> None:
        super().__init__()

        self.p = p

    def forward(self, x):
        # Only perform batch element dropping during training
        if not self.training: return x

        b, *_ = x.shape

        # Assemble the drop mask with broadcastable shape
        drop = torch.rand(b, device = x.device).le(self.p)
        drop = enlarge_as(drop, x)

        # Mask the batched input via mask multiplication
        return drop * x