import torch.nn as nn
from torch import Tensor

from einops import rearrange

from ..utils.misc import exists
from ..utils.misc import default

from typing import Tuple, Optional, Union

class LinearAdapter(nn.Module):
    '''
        Adapter needed by an Attention Layer to
        adjust its behaviour to pure sequence-like
        inputs.
    '''

    def __init__(
        self,
        qry_dim : int,
        emb_dim : int,        
        key_dim : Optional[int] = None,
        val_dim : Optional[int] = None,
    ) -> None:
        super().__init__()

        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, key_dim)

        self.to_q = nn.Linear(qry_dim, emb_dim, bias = False)
        self.to_k = nn.Linear(key_dim, emb_dim, bias = False)
        self.to_v = nn.Linear(val_dim, emb_dim, bias = False)

        self.from_q = nn.Linear(emb_dim, qry_dim, bias = False)

    def forward(
        self,
        qry : Tensor,
        key : Optional[Tensor] = None,
        val : Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        if exists(key) and exists(val):
            return self._proj_in(qry, key, val)
        else:
            return self._proj_out(qry)


    def _proj_in(
        self,
        qry : Tensor,
        key : Tensor,
        val : Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        
        q = self.to_q(qry)
        k = self.to_k(key)
        v = self.to_v(val)

        return q, k, v
    
    def _proj_out(self, qry : Tensor) -> Tensor:
        return self.from_q(qry)
    
class ConvLinearAdapter(nn.Module):
    '''
        Adapter needed by an Attention Layer to
        adjust its behaviour to image-like inputs
    '''
    def __init__(
        self,
        qry_dim : int,
        emb_dim : int,    
        key_dim : Optional[int] = None,
        val_dim : Optional[int] = None,
    ) -> None:
        super().__init__()

        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, key_dim)

        self.to_q = nn.Conv2d(qry_dim, emb_dim, 1, bias = False)
        self.to_k = nn.Linear(key_dim, emb_dim, bias = False)
        self.to_v = nn.Linear(val_dim, emb_dim, bias = False)

        self.from_q = nn.Conv2d(emb_dim, qry_dim, 1, bias = False)

    def forward(
        self,
        qry : Tensor,
        key : Optional[Tensor] = None,
        val : Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        
        if exists(key) and exists(val):
            return self._proj_in(qry, key, val)
        else:
            return self._proj_out(qry)

    def _proj_in(
        self,
        qry : Tensor,
        key : Tensor,
        val : Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Store q shape for out projection
        *_, self.h, self.w = qry.shape

        q = self.to_q(qry)
        k = self.to_k(key)
        v = self.to_v(val)

        q = rearrange(q, 'b c h w -> b (h w) c').contiguous()

        return q, k, v
    
    def _proj_out(self, qry : Tensor) -> Tensor:
        if not hasattr(self, 'h'):
            raise ValueError('Cannot call adapt._out before the _in method')
        
        qry = rearrange(qry, 'b (h w) c -> b c h w', h = self.h, w = self.w).contiguous()

        return self.from_q(qry)

class ConvAdapter(nn.Module):
    '''
        Adapter needed by an Attention Layer to
        adjust its behaviour to image-like inputs
    '''
    def __init__(
        self,
        qry_dim : int,
        emb_dim : int,    
        key_dim : Optional[int] = None,
        val_dim : Optional[int] = None,
    ) -> None:
        super().__init__()

        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, key_dim)

        self.to_q = nn.Conv2d(qry_dim, emb_dim, 1, bias = False)
        self.to_k = nn.Conv2d(key_dim, emb_dim, 1, bias = False)
        self.to_v = nn.Conv2d(val_dim, emb_dim, 1, bias = False)

        self.from_q = nn.Conv2d(emb_dim, qry_dim, 1, bias = False)

    def forward(
        self,
        qry : Tensor,
        key : Optional[Tensor] = None,
        val : Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        
        if exists(key) and exists(val):
            return self._proj_in(qry, key, val)
        else:
            return self._proj_out(qry)

    def _proj_in(
        self,
        qry : Tensor,
        key : Tensor,
        val : Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Store q shape for out projection
        *_, self.h, self.w = qry.shape

        q = self.to_q(qry)
        k = self.to_k(key)
        v = self.to_v(val)

        q = rearrange(q, 'b c h w -> b (h w) c').contiguous()
        k = rearrange(k, 'b c h w -> b (h w) c').contiguous()
        v = rearrange(v, 'b c h w -> b (h w) c').contiguous()

        return q, k, v
    
    def _proj_out(self, qry : Tensor) -> Tensor:
        if not hasattr(self, 'h'):
            raise ValueError('Cannot call adapt._out before the _in method')
        
        qry = rearrange(qry, 'b (h w) c -> b c h w', h = self.h, w = self.w).contiguous()

        return self.from_q(qry)
