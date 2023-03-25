import torch
from math import pi
from torch import Tensor

from torch.special import expm1

from typing import Tuple, Callable

def linear_beta_schedule(timesteps : int) -> Tensor:
    scale = 1000 / timesteps

    beta_start = scale * 0.0001
    beta_end   = scale * 0.02

    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps : int, s : float = 0.008, **kwargs) -> Tensor:
    t = torch.linspace(0, timesteps, timesteps + 1, dtype = torch.float64) / timesteps
    
    alphas_cumprod = torch.cos((t + s) / (1 + s) * pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(
        timesteps : int,
        start : int = -3,
        end   : int = 3,
        tau   : int = 1,
        clamp_min : float = 0,
        **kwargs
    ) -> Tensor:
    
    t = torch.linspace(0, timesteps, timesteps + 1, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, clamp_min, 0.999)

def build_beta_schedule(timesteps : int, kind : str, **kwargs) -> Tensor:
    if   kind == 'linear' : return linear_beta_schedule (timesteps)
    elif kind == 'cosine' : return cosine_beta_schedule (timesteps, **kwargs)
    elif kind == 'sigmoid': return sigmoid_beta_schedule(timesteps, **kwargs)
    else:
        raise ValueError(f'Unknown beta schedule: {kind}')
    
def linear_beta_log_snr(timesteps : Tensor) -> Tensor:
    return -torch.log(expm1(1e-4 + 10 * (timesteps ** 2))).clamp(min = 1e-20)

def cosine_alpha_log_snr(timesteps : Tensor, s : float = 0.008, **kwargs) -> Tensor:
    return -torch.log((torch.cos((timesteps + s) / (1 + s) * pi * 0.5) ** -2) - 1).clamp(min = 1e-20)

def log_snr_to_alpha(log_snr : Tensor) -> Tuple[Tensor, Tensor]:
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

def build_noise_schedule(schedule : str) -> Callable:
    if   schedule == 'linear': return linear_beta_log_snr
    elif schedule == 'cosine': return cosine_alpha_log_snr
    else:
        raise ValueError('Unknown noise schedule: {schedule}')