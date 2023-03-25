import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
from einops import einsum

from ..utils.misc import exists
from ..utils.misc import default

from .adapter import LinearAdapter

from typing import Tuple, Optional

def embed(seq : Tensor, pos : Optional[Tensor]) -> Tensor:
    return (seq + pos) if exists(pos) else seq

class EfficientAttention(nn.Module):
    '''
        Implementation of Efficient Attention from the paper:
        "Self-attention Does Not Need O(n²) Memory".

        Code is heavily based on:
        https://github.com/lucidrains/memory-efficient-attention-pytorch
    '''

    def __init__(
        self,
        input_dim : int,
        embed_dim : int,
        num_heads : int,
        qry_chunk : int = 512,
        key_chunk : int = 1024,
        dropout   : float = 0.,
        qry_embed : Optional[Tensor] = None,
        key_embed : Optional[Tensor] = None,
        qkv_adapt : Optional[nn.Module] = None,
        pre_norm  : bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads

        self.qry_chunk = qry_chunk
        self.key_chunk = key_chunk
        self.qry_embed = qry_embed
        self.key_embed = key_embed

        self.dropout = dropout
        self.pre_norm = pre_norm

        Adapter = default(qkv_adapt, LinearAdapter)
        self.qkv_adapt = Adapter(input_dim, embed_dim, **kwargs)

        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        qry : Tensor,
        key : Optional[Tensor] = None,
        val : Optional[Tensor] = None,
        qry_chunk : Optional[int] = None,
        key_chunk : Optional[int] = None,
        attn_mask : Optional[Tensor] = None,
        attn_bias : Optional[Tensor] = None,
        key_embed : Optional[Tensor] = None,
        qry_embed : Optional[Tensor] = None,
    ) -> Tensor:
        qry_chunk = default(qry_chunk, self.qry_chunk)
        key_chunk = default(key_chunk, self.key_chunk)

        key = default(key, qry)
        val = default(val, key)

        qry_embed = default(qry_embed, self.qry_embed)
        key_embed = default(key_embed, self.key_embed)

        h = self.num_heads

        # When adapter is called with full arguments it
        # triggers the in adapter
        qry, key, val = self.qkv_adapt(qry, key, val)

        qry = self.norm(qry) if self.pre_norm else qry

        qry = embed(qry, qry_embed)
        key = embed(key, key_embed)

        # Create multiple attention heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h).contiguous(), (qry, key, val))

        attn = self.attn_fn(q, k, v,
            qry_chunk = qry_chunk,
            key_chunk = key_chunk,
            attn_mask = attn_mask,
            attn_bias = attn_bias,
            dropout = self.dropout,
        )

        out = rearrange(attn, 'b h n d -> b n (h d)').contiguous()
        
         # Add residual connection and normalization
        out += qry

        out = out if self.pre_norm else self.norm(out)

        # When called with a single argument it triggers
        # the output adapter
        return self.qkv_adapt(out)

    def attn_fn(
        self,
        qry : Tensor,
        key : Tensor,
        val : Tensor,
        qry_chunk : int = 512,
        key_chunk : int = 1024,
        attn_mask : Optional[Tensor] = None,
        attn_bias : Optional[Tensor] = None,
        dropout : float = 0.,
    ) -> Tensor:
        '''
            Compute the attention function in memory efficient fashion
            by splitting the query and key|values into chunks and
            processing each chunk by so limiting the memory footprint.
        '''

        b, h, n, d = qry.shape

        qry *= d ** -0.5

        # Chunk all the inputs along the sequence length dimension
        q_chunks = qry.split(qry_chunk, dim = -2)
        k_chunks = key.split(key_chunk, dim = -2)
        v_chunks = val.split(key_chunk, dim = -2)

        mask_chunks = attn_mask.split(key_chunk, dim = -1) if exists(attn_mask)\
                        else ((None,) * len(k_chunks))
        
        if exists(attn_bias):
            *_, i, j = attn_bias.shape
            bias_chunks = attn_bias.split(qry_chunk, dim = -2)
            bias_chunks = list(map(lambda t: t.split(key_chunk, dim = -1), bias_chunks))
        else:
            bias_chunks = None

        # Iterate through all the chunks and accumulate the (partial) attention 
        attn = []

        for q_idx, q_chunk in enumerate(q_chunks):
            # Reset accumulators
            attn_chunk = []
            exp_qkv = []
            max_qkv = []

            for k_idx, (k_chunk, v_chunk, m_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
                
                b_chunk = bias_chunks[q_idx][k_idx] if exists(attn_bias) else None

                _attn_chunk, _exp_qkv, _max_qkv = self._sum_qkv_chunk(
                    q_chunk, k_chunk, v_chunk,
                    qk_idxs = (q_idx, k_idx),
                    attn_mask = m_chunk,
                    attn_bias = b_chunk,
                    dropout = dropout,
                )

                attn_chunk.append(_attn_chunk)
                exp_qkv.append(_exp_qkv)
                max_qkv.append(_max_qkv)

            # Stack all the key chuncks together
            attn_chunk = torch.stack(attn_chunk, dim = -1)
            exp_qkv = torch.stack(exp_qkv, dim = -1)
            max_qkv = torch.stack(max_qkv, dim = -1)

            # Compute the global maximum values across all the chuncks
            # and renormalize the running computations before storing
            # results for the actual attention output
            global_max = max_qkv.amax(dim = -1, keepdim = True)
            renorm_val = (max_qkv - global_max).exp().detach()

            exp_qkv *= renorm_val
            attn_chunk *= rearrange(renorm_val, '... c -> ... 1 c').contiguous()

            # Obtained attention by summing along the chunk dimension
            qkv = attn_chunk.sum(dim = -1) / (1e-8 + exp_qkv.sum(dim = -1, keepdim = True))

            attn.append(qkv)

        # Return the attention by contatenating all the queries together
        return torch.cat(attn, dim = -2)


    def _sum_qkv_chunk(
        self,
        qry : Tensor,
        key : Tensor,
        val : Tensor,
        qk_idxs : Tuple[int, int],
        attn_mask : Optional[Tensor] = None,
        attn_bias : Optional[Tensor] = None,
        dropout : float = 0.,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        '''
            Consume a chunck of the attention qkv by keeping track
            of the running sums of qk and qkv, with their max value
            for numerical stability.
        '''

        attn_bias = default(attn_bias, 0.)

        # q_start, k_start = qk_idxs
        # *_, q_chunk, _ = qry.shape
        # *_, k_chunk, _ = key.shape

        # Compute the (chunk) query-key overlap
        qk : Tensor = einsum(qry, key, 'b h i d, b h j d -> b h i j') + attn_bias

        if exists(attn_mask):
            mask_value = -torch.finfo(qk.dtype).max
            qk = qk.masked_fill(~attn_mask, mask_value)

        # Improve numerical stability by subtracting the maximum value
        # before computing the exponentials
        qk_max = qk.amax(dim = -1, keepdim = True).detach()
        qk -= qk_max

        exp_qk = F.dropout(qk.exp(), p = dropout, training = self.training)

        attn = einsum(exp_qk, val, 'b h i j, b h j d -> b h i d')

        return attn, exp_qk.sum(dim = -1), qk_max.squeeze()