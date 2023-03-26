import torch
import torch.nn as nn
from torch import Tensor

from einops import rearrange

from ..utils.misc import default
from ..utils.misc import BatchDrop

from typing import Optional

class TimeEmbedding(nn.Module):
    '''
        Embedding for time-like data used by diffusion models.
    '''

    def __init__(
        self,
        emb_dim : int,
        base : int = 10000
    ) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.base = base

    def forward(self, time : Tensor) -> Tensor:
        time = torch.atleast_1d(time)
        bs = len(time)

        half_dim = self.emb_dim // 2        
        emb_time = torch.empty((bs, self.emb_dim), device = time.device)

        pos_n = torch.arange(half_dim, device = time.device)
        inv_f = 1. / (self.base ** (pos_n / (half_dim - 1)))

        emb_v = torch.outer(time, inv_f)

        emb_time[..., 0::2] = emb_v.sin()
        emb_time[..., 1::2] = emb_v.cos()

        return emb_time

class SineEmbedding1D(nn.Module):
    '''
        1-D implementation of the sinusoidal positional embedding.
    '''

    def __init__(
        self,
        max_len : int,
        base : int = 10000,    
    ) -> None:
        super().__init__()
    
        self.base = base
        self.max_len = max_len

    def forward(self, inp : Tensor) -> Tensor:
        '''
            Add sinusoidal positional embedding to input tensor.
        '''

        bs, n, d, device = *inp.shape, inp.device

        if not hasattr(self, 'encoding'):
            self.encoding = torch.empty((n, d), device = device)

            pos_n = torch.arange(n)
            pos_k = torch.arange(0, d, 2)
            inv_f = 1. / (self.base ** (pos_k / d))

            emb_v = torch.outer(pos_n, inv_f)

            self.encoding[:, 0::2] = emb_v.sin()
            self.encoding[:, 1::2] = emb_v.cos()

        return inp + self.encoding

class SineEmbedding2D(nn.Module):
    '''
        2-D Implementation of the Sine Positional Encodings that
        suits the Transformer semantic [batch_size, embed_dim, (H*W)]
    '''

    def __init__(
        self,
        max_len  : int,
        base : float = 10000
    ) -> None:
        super().__init__()

        self.max_len = max_len
        self.base = base

    def forward(self, inp : Tensor) -> Tensor:
        bs, *_, h, w = inp.shape

        ngrid = torch.ones((bs, h, w, 1), device = inp.device)

        pos_x = torch.cumsum(ngrid, 2)
        pos_y = torch.cumsum(ngrid, 1)

        pos_n = torch.arange(self.max_len, device = inp.device)
        inv_f = 1. / (self.base ** (pos_n / self.max_len))

        pos_x = pos_x * inv_f
        pos_y = pos_y * inv_f 

        # Embeddings now have shape [bs, h, w, max_len], we stack them on the last dimension
        # and flatten so the ending result is a tensor of shape:
        # [bs, h, w, max_len / 2, 2] -> [bs, h, w, max_len]
        emb_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim = -1).flatten(3)
        emb_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim = -1).flatten(3)

        # Finally we asseble the two coordinates back together along the last dimension,
        # we get an embedding tensor of shape: [bs, h, w, max_len * 2]
        # * NOTE: It is important that max_len is actually equal to HALF the embedding
        # *       dimension of the transformer model, so that all dimensions match.
        emb = torch.cat([emb_x, emb_y], dim = -1)

        # We impose the convention of dimension semantic [batch_size, embed_dim, *]
        return rearrange(emb, 'b h w e -> b (h w) e').contiguous()
        
class FeatEmbedder(nn.Module):
    '''
        This module takes a 1-dimensional signal and embeds it into
        a space of given dimensionality via 1D Convolutions
    '''

    def __init__(
        self,
        patch_len : int,
        embed_dim : int,
        inp_chnnl : int = 1,
        stride : Optional[int] = None
    ) -> None:
        super().__init__()

        stride = default(stride, patch_len)

        self.patch_len = patch_len
        self.embed_dim = embed_dim

        self.embed = nn.Conv1d(inp_chnnl, embed_dim, kernel_size = patch_len, stride = stride)

    def forward(self, seq : Tensor) -> Tensor:
        # Verify input shape to be: [batch_size, channel, seq_len]
        b, c, n = seq.shape

        # Compute sequence embedding. At this stage we have a
        # tensor of shape [batch_size, embed_dim, new_len]
        out = self.embed(seq)

        # Rearrange to match the transformers semantics of dimension:
        # [batch_size, new_len, embed_dim]
        out = rearrange(out, 'b c n -> b n c').contiguous()

        return out
    
class ClassEmbedder(nn.Module):
    def __init__(
        self,
        emb_dim : int,
        cls_dim : int,
        p_dropb : float = .1
    ) -> None:
        super().__init__()

        self.embed = nn.Embedding(cls_dim, emb_dim)
        self.dropb = BatchDrop(p_dropb)

    def forward(self, batch):
        emb : Tensor = self.embed(batch)

        return self.dropb(emb)

