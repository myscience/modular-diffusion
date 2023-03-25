import torch
from torch import Tensor
from torch.nn import Module
from .diffusion import Diffusion

from .module.utils.misc import default

from typing import Optional

class ElucidatedDiffusion(Diffusion):
    '''
        Denoising Diffusion Probabilistic Model as introduced in:
        "Elucidating the Design Space of Diffusion-Based Generative
        Models", Kerras et al. (2022) (https://arxiv.org/pdf/2206.00364)
    '''

    def __init__(
        self,
        model: Module,
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

        # Controls for timesteps and schedule generation
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho_schedule = rho_schedule

        # Controls for training
        self.lognorm_mean = lognorm_mean
        self.lognorm_std  = lognorm_std
        self.sigma_data   = sigma_data

        self.log_img_key = f'Eludidated Diffusion - {ode_solver}'

        self.norm_forward  = lambda x : 2.  * x - 1.
        self.norm_backward = lambda x : 0.5 * (1 + x.clamp(-1., 1.))

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

