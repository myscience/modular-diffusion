import torch.nn as nn
from torch import Tensor
from torch.nn.functional import interpolate

from ..utils.misc import default
from ..utils.misc import enlarge_as

from typing import List, Optional

def Upsample(dim_in, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim_in, default(dim_out, dim_in), 3, padding = 1)
    )

def Downsample(dim_in, dim_out = None):
    return nn.Conv2d(dim_in, default(dim_out, dim_in), 4, 2, 1)

class LinRes(nn.Module):
    '''
        Linear Residual Block. It is composed of two
        linear layers and outputs the addition of the
        output of those layers with the residual path
    '''

    def __init__(
        self,
        inp_dim : int,
        hid_dim : int,
        dropout : float = 0.   
    ) -> None:
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(inp_dim, hid_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, inp_dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(inp_dim)

    def forward(self, inp : Tensor) -> Tensor:
        out = inp + self.seq(inp)

        return self.norm(out)

class ConvRes(nn.Module):
    '''
        Convolutional Residual Block. It is composed of two
        convolutional layers and output the addition of the
        output of those convolutions with the residual path
    '''

    def __init__(
        self,
        inp_chn : int,
        out_chn : Optional[int] = None,
        hid_chn : Optional[int] = None,

        stride : int = 1,
    ) -> None:
        super().__init__()

        out_chn = default(out_chn, inp_chn)
        hid_chn = default(hid_chn, out_chn)

        self.seq = nn.Sequential(
            nn.Conv2d(inp_chn, hid_chn, kernel_size = 1, bias = False),
            nn.BatchNorm2d(hid_chn),
            nn.ReLU(inplace = True),
            nn.Conv2d(hid_chn, out_chn, kernel_size = 3, padding = 1, stride = stride, bias = False),
            nn.BatchNorm2d(out_chn),
        )

        proj_ker = 1 if stride == 1 else 3
        self.proj = nn.Conv2d(inp_chn, out_chn, proj_ker, padding = 1, stride = stride)

        self.actv = nn.ReLU(inplace = True)

    def forward(self, inp : Tensor) -> Tensor:
        out = self.proj(inp) + self.seq(inp)

        return self.actv(out)
    
class ConvTimeRes(nn.Module):
    '''
        Convolutional Residual Block with time embedding
        injection support, used by Diffusion Models. It is
        composed of two convolutional layers with normalization.
        The time embedding signal is injected between the two
        convolutions and is added to the input to the second one.
    '''

    def __init__(
        self,
        inp_dim : int,
        out_dim : Optional[int] = None,
        hid_dim : Optional[int] = None,
        ctx_dim : Optional[int] = None,
        num_group : int = 8,
    ) -> None:
        super().__init__()

        out_dim = default(out_dim, inp_dim)
        hid_dim = default(hid_dim, out_dim)
        ctx_dim = default(ctx_dim, out_dim)

        self.time_emb = nn.Sequential(
            nn.SiLU(inplace = False),
            nn.Linear(ctx_dim, hid_dim),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(inp_dim, hid_dim, kernel_size = 3, padding = 1),
            nn.GroupNorm(num_group, hid_dim),
            nn.SiLU(inplace = False),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hid_dim, out_dim, kernel_size = 3, padding = 1),
            nn.GroupNorm(num_group, out_dim),
            nn.SiLU(inplace = False),
        )

        self.skip = nn.Conv2d(inp_dim, out_dim, 1) if inp_dim != out_dim else nn.Identity()

    def forward(
        self,
        inps : Tensor,
        time : Tensor,
    ) -> Tensor:
        
        # Perform first convolution block
        h = self.conv1(inps)

        # Add embedded time signal with appropriate
        # broadcasting to match image-like tensors
        time = self.time_emb(time)
        h += enlarge_as(time, h)

        h = self.conv2(h)

        return self.skip(inps) + h