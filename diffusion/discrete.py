import torch
from torch import Tensor
from torch.nn import Module
from .diffusion import Diffusion

from .module.utils.misc import default

from einops import reduce
from einops import rearrange
from functools import partial
from typing import Optional

class BitDiffusion(Diffusion):
    '''
        Discrete Denoising Diffusion Probabilistic Model as introduced in:
        "Analog Bits: Generating Discrete Data Using Diffusion Models with
        Self-Conditioning", Chen et al. (2023) - Hinton's Group.
    '''

    def __init__(
        self,
        model: Module,
        num_bits : int = 1,
        data_type : str = 'int',
        bit_scale : float = 1.,
        sigma_min : float = 0.002,
        sigma_max : float = 80,
        ode_solver : str = 'heun_sde',
        rho_schedule : float = 7,
        lognorm_mean : float = -1.2,
        lognorm_std  : float = +1.2,
        sigma_data : float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            ode_solver = ode_solver,
            **kwargs,
        )

        self.num_bits = num_bits
        self.bit_scale = bit_scale

        # Controls for timesteps and schedule generation
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho_schedule = rho_schedule

        # Controls for training
        self.lognorm_mean = lognorm_mean
        self.lognorm_std  = lognorm_std
        self.sigma_data   = sigma_data

        self.log_img_key = f'Bit Diffusion - {ode_solver}'

        if data_type == 'int':
            data2bit = self.int2bit
            bit2data = self.bit2int
        elif data_type == 'float':
            data2bit = self.float2bit
            bit2data = self.bit2float
        else:
            raise ValueError(f'Unsupported date type {data_type}')

        self.norm_forward  = partial(data2bit, nbits = num_bits, scale = bit_scale)
        self.norm_backward = partial(bit2data, nbits = num_bits)

    # * Functions that define what model actually predicts
    def c_skip(self, sigma : Optional[Tensor] = None) -> Tensor:
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma : Optional[Tensor] = None) -> Tensor:
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5
    
    def c_in(self, sigma : Optional[Tensor] = None) -> Tensor:
        return (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma : Optional[Tensor] = None) -> Tensor:
        return torch.log(sigma.clamp(min = 1e-20)) * 0.25

    # * Functions that define model training
    def loss_weight(self, sigma : Tensor) -> Tensor:
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2
    
    def get_noise(self, batch_size : int) -> Tensor:
        eps = torch.randn((batch_size,), device = self.device)
        return (self.lognorm_mean + self.lognorm_std * eps).exp()
    
    # * Functions that define sampling strategy
    def get_timesteps(self, num_steps : int, rho : Optional[int] = None) -> Tensor:
        rho = default(rho, self.rho_schedule)

        inv_rho = 1 / rho

        tramp = torch.linspace(0, 1, num_steps, device = self.device)
        i_max = self.sigma_max ** inv_rho
        i_min = self.sigma_min ** inv_rho

        sigma = (i_max + tramp * (i_min - i_max)) ** rho

        return sigma

    def get_schedule(self, t : Tensor, **kwargs) -> Tensor:
        return t

    def get_scaling(self, t : Tensor, **kwargs) -> Tensor:
        return torch.ones_like(t)

    @classmethod
    def int2bit(cls, decs : Tensor, nbits : int = 8, scale : float = 1.) -> Tensor:
        '''
            Convert input (int) tensor x (values in [0, 255])
            to analog bits in [-1, 1].
        '''
        device = decs.device

        decs = decs.clamp(min = 0, max = 255).long()

        # Build the bitmask needed for decimal-to-bit conversion
        mask = 2 ** torch.arange(nbits - 1, -1, -1, device = device, dtype = torch.long)

        mask = rearrange(mask, 'd -> d 1 1').contiguous()
        decs = rearrange(decs, 'b c h w -> b c 1 h w').contiguous()

        # Get the analog bits
        bits = ((decs & mask) != 0).float()
        bits = rearrange(bits, 'b c d h w -> b (c d) h w').contiguous()

        return (bits * 2 - 1) * scale
    
    @classmethod
    def float2bit(cls, decs : Tensor, nbits : int = 8, scale : float = 1.) -> Tensor:
        '''
            Convert input (float) tensor x (values in [0, 1])
            to analog bits in [-1, 1].
        '''
        device = decs.device

        decs = (decs * 255).clamp(0, 255).long()

        # Build the bitmask needed for decimal-to-bit conversion
        mask = 2 ** torch.arange(nbits - 1, -1, -1, device = device, dtype = torch.long)

        mask = rearrange(mask, 'd -> d 1 1').contiguous()
        decs = rearrange(decs, 'b c h w -> b c 1 h w').contiguous()

        bits = ((decs & mask) != 0).float()
        bits = rearrange(bits, 'b c d h w -> b (c d) h w').contiguous()

        return (bits * 2 - 1) * scale
    
    @classmethod
    def bit2int(cls, bits : Tensor, nbits : int = 8) -> Tensor:
        '''
        '''
        device = bits.device

        bits = (bits > 0).int()
        mask = 2 ** torch.arange(nbits - 1, -1, -1, device = device, dtype = int)

        mask = rearrange(mask, 'd -> d 1 1').contiguous()
        bits = rearrange(bits, 'b (c d) h w -> b c d h w', d = nbits).contiguous()

        decs = reduce(bits * mask, 'b c d h w -> b c h w', 'sum').contiguous()

        return decs.clamp(0, 255)

    @classmethod
    def bit2float(cls, bits : Tensor, nbits : int = 8) -> Tensor:
        '''
        '''
        
        decs = cls.bit2int(bits, nbits)

        return (decs / 255).clamp(0., 1.)