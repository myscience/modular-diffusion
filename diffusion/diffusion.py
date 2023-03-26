import yaml
import torch
import torch.nn as nn
from torch import Tensor
from abc import abstractclassmethod

from torch.nn.functional import l1_loss
from torch.nn.functional import mse_loss
from torchvision.utils import make_grid

from math import sqrt, log, expm1

from torch.optim import AdamW

from .module.unet import UNet

from .module.utils.misc import exists
from .module.utils.misc import default
from .module.utils.misc import enlarge_as
from .module.utils.misc import groupwise

from random import random
from einops import reduce

from pytorch_lightning import LightningModule

from tqdm.auto import tqdm

from typing import Tuple, Callable, Dict, Optional

class Diffusion(LightningModule):
    '''
        Implementation of a Denoising Diffusion Probabilistic Model (DDPM).
        This implementation follows a modular designed inspired by Karras et
        al. (2022) "Elucidating the Design Space of Diffusion-Based Generative
        Models" (https://arxiv.org/pdf/2206.00364). 
    '''

    @classmethod
    def from_conf(cls, path : str) -> 'Diffusion':
        '''
            Initialize the Diffusion model from a YAML configuration file.
        '''

        with open(path, 'r') as f:
            conf = yaml.safe_load(f)

        # Store the configuration file
        cls.conf = conf

        net_par = conf['MODEL']
        dif_par = conf['DIFFUSION']

        # Grab the batch size for precise metric logging
        cls.batch_size = conf['DATASET']['batch_size']

        # Initialize the network
        net = UNet(**net_par)

        return cls(net, **dif_par)
    
    def __init__(
        self, 
        model : nn.Module,
        img_size : int = 32,
        loss_type : str = 'L2',
        self_cond : bool = True,
        ode_solver : str = 'ddim',
        time_delta : float = 0.,
        sample_steps : int = 35,
        norm_forward  : Optional[Callable] = None,
        norm_backward : Optional[Callable] = None,
        data_key : str = 'smap',
        ctrl_key : Optional[str] = None,
    ) -> None:
        super().__init__()

        assert ode_solver in ('ddim', 'dpm++', 'heun', 'heun_sde')

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.model = model

        self.loss_type = loss_type
        self.self_cond = self_cond
        self.ode_solver = ode_solver
        self.time_delta = time_delta
        self.sample_steps = sample_steps

        self.img_size = img_size
        self.data_key = data_key
        self.ctrl_key = ctrl_key

        self.norm_forward  = default(norm_forward,  lambda x : x)
        self.norm_backward = default(norm_backward, lambda x : x)

        self.log_img_key = f'Diffusion - {ode_solver}'

        # Each diffusion model should overwrite its external control
        # embedding module based on the nature of the control itself 
        self.ctrl_emb : Optional[nn.Module] = None
        self.null_ctrl = lambda ctrl : torch.zeros_like(ctrl)

        self.save_hyperparameters()
    
    @property
    def criterion(self) -> Callable:
        if   self.loss_type == 'L1': return l1_loss
        elif self.loss_type == 'L2': return mse_loss
        else:
            raise ValueError(f'Unknown objective: {self.loss_type}')

    @property
    def sampler(self) -> Callable:
        solver = self.ode_solver
        if   solver == 'heun' :     return self.heun
        elif solver == 'dpm++':     return self.dpmpp
        elif solver == 'ddim' :     return self.ddim
        elif solver == 'heun_sde':  return self.heun_sde
        elif solver == 'dpm++_sde': return self.dpmpp_sde
        else:
            raise ValueError(f'Unknow sampler {solver}')

    @property
    def device(self):
        return next(self.model.parameters()).device
    
    # * Functions that define what model actually predicts
    @abstractclassmethod
    def c_skip(self, sigma : Optional[Tensor] = None) -> Tensor:
        pass

    @abstractclassmethod
    def c_out(self, sigma : Optional[Tensor] = None) -> Tensor:
        pass

    @abstractclassmethod
    def c_in(self, sigma : Optional[Tensor] = None) -> Tensor:
        pass

    @abstractclassmethod
    def c_noise(self, sigma : Optional[Tensor] = None) -> Tensor:
        pass

    # * Functions that define model training
    @abstractclassmethod
    def loss_weight(self, sigma : Tensor) -> Tensor:
        pass

    @abstractclassmethod
    def get_noise(self, batch_size : int) -> Tensor:
        pass

    # * Functions that define sampling strategy
    @abstractclassmethod
    def get_timesteps(self, num_steps : int, **kwargs) -> Tensor:
        pass

    @abstractclassmethod
    def get_schedule(self, t : Tensor, **kwargs) -> Tensor:
        pass

    @abstractclassmethod
    def get_scaling(self, t : Tensor, **kwargs) -> Tensor:
        pass

    def predict(
        self,
        x_t : Tensor,
        sig : Tensor,
        x_c  : Optional[Tensor] = None,
        ctrl : Optional[Tensor] = None,
        clamp : bool = False,
    ) -> Tensor:
        '''
            Apply the backbone model to come up with a prediction, the
            nature of which depends on the diffusion objective (can either
            be noise|x_start|v prediction).
        '''

        bs, *_, device = x_t.shape, x_t.device

        if isinstance(sig, float): sig = torch.full((bs,), sig, device = device)

        # Inject appropriate noise value to images
        p_sig = enlarge_as(sig, x_t)
        x_sig = self.c_in(p_sig) * x_t
        t_sig = self.c_noise(sig)

        # Use the model to come up with a (hybrid) prediction the nature of
        # which depends on the implementation of the various c_<...> terms
        # so that the network can either predict the noise (eps) or the
        # input directly (better when noise is large!)
        out : Tensor = self.model(x_sig, t_sig, x_c = x_c, ctrl = ctrl)
        out : Tensor = self.c_skip(p_sig) * x_t + self.c_out(p_sig) * out

        if clamp: out = out.clamp(-1., 1.)

        return out
    
    @torch.no_grad()
    def follow(self, *args, ctrl : Optional[Tensor] = None, guide : float = 1., **kwargs):
        '''
            Implements Classifier-Free guidance as introduced
            in Ho & Salimans (2022). 
        '''
        if not exists(ctrl) or guide == 1:
            return self.predict(*args, ctrl = ctrl, **kwargs)

        # Get the unconditioned & conditioned predictions
        null = self.predict(*args, ctrl = self.null_ctrl(ctrl), **kwargs)
        cond = self.predict(*args, ctrl = ctrl, **kwargs)

        # Compose the classifier-free prediction
        return null + guide * (cond - null)
    
    @torch.no_grad()
    def forward(
        self,
        num_imgs : int = 4,
        num_steps : Optional[int] = None,
        ode_solver : Optional[str] = None,
        norm_undo : Optional[Callable] = None,
        ctrl : Optional[Tensor] = None,
        use_x_c : Optional[bool] = None,
        guide : float = 1.,
        **kwargs,
    ) -> Tensor:
        '''
            Sample images using a given sampler (ODE Solver)
            from the trained model. 
        '''

        use_x_c = default(use_x_c, self.self_cond)
        num_steps = default(num_steps, self.sample_steps)
        norm_undo = default(norm_undo, self.norm_backward)
        self.ode_solver = default(ode_solver, self.ode_solver)

        timestep = self.get_timesteps(num_steps)
        schedule = self.get_schedule(timestep)
        scaling  = self.get_scaling (timestep)

        # schedule = repeat(schedule, '... -> b ...', b = num_imgs)
        # scaling  = repeat(scaling , '... -> b ...', b = num_imgs)

        # Encode the condition using the sequence encoder
        ctrl = self.ctrl_emb(ctrl)[:num_imgs] if exists(ctrl) else ctrl

        shape = (num_imgs, self.model.channels, *self.img_size)

        x_0 = self.sampler(
            shape,
            schedule,
            scaling,
            ctrl = ctrl,
            use_x_c = use_x_c,
            guide = guide,
            **kwargs
        )

        return norm_undo(x_0)
    
    def compute_loss(
        self,
        x_0 : Tensor,
        ctrl : Optional[Tensor] = None,
        use_x_c : Optional[bool] = None,     
        norm_fn : Optional[Callable] = None,
    ) -> Tensor:

        use_x_c = default(use_x_c, self.self_cond)
        norm_fn = default(norm_fn, self.norm_forward)

        # Encode the condition using the sequence encoder
        ctrl = self.ctrl_emb(ctrl) if exists(ctrl) else ctrl

        # Normalize input images
        x_0 = norm_fn(x_0)

        bs, *_ = x_0.shape

        # Get the noise and scaling schedules
        sig = self.get_noise(bs)

        # NOTE: What to do with the scaling if present?
        # scales = self.get_scaling()

        eps = torch.randn_like(x_0)
        x_t = x_0 + enlarge_as(sig, x_0) * eps # NOTE: Need to consider scaling here!

        x_c = None

        # Use self-conditioning with 50% dropout
        if use_x_c and random() < 0.5:
            with torch.no_grad():
                x_c = self.predict(x_t, sig, ctrl = ctrl)
                x_c.detach_()

        x_p = self.predict(x_t, sig, x_c = x_c, ctrl = ctrl)

        # Compute the reconstruction loss
        loss = self.criterion(x_p, x_0, reduction = 'none')
        loss : Tensor = reduce(loss, 'b ... -> b', 'mean')

        # Add loss weight
        loss *= self.loss_weight(sig)
        return loss.mean()

    # * Lightning Module functions
    def training_step(self, batch : Dict[str, Tensor], batch_idx : int) -> Tensor:
        # Extract the starting images from data batch
        x_0  = batch[self.data_key]
        ctrl = batch[self.ctrl_key] if exists(self.ctrl_key) else None

        loss = self.compute_loss(x_0, ctrl = ctrl)

        self.log_dict({'train_loss' : loss}, logger = True, on_step = True, sync_dist = True)

        return loss
    
    def validation_step(self, batch : Dict[str, Tensor], batch_idx : int) -> Tensor:
        # Extract the starting images from data batch
        x_0  = batch[self.data_key]
        ctrl = batch[self.ctrl_key] if exists(self.ctrl_key) else None

        loss = self.compute_loss(x_0, ctrl = ctrl)

        self.log_dict({'val_loss' : loss}, logger = True, on_step = True, sync_dist = True)

        return x_0, ctrl

    @torch.no_grad()
    def validation_epoch_end(self, val_outs : Tuple[Tensor, ...]) -> None:
        '''
            At the end of the validation cycle, we inspect how the denoising
            procedure is doing by sampling novel images from the learn distribution.
        '''

        # Collect the input shapes
        (x_0, ctrl), *_ = val_outs

        # Produce 8 samples and log them
        imgs = self(
                num_imgs = 8,
                ctrl = ctrl,
                verbose = False,
            )
        
        assert not torch.isnan(imgs).any(), 'NaNs detected in imgs!'

        imgs = make_grid(imgs, nrow = 4)

        # Log images using the default TensorBoard logger
        self.logger.experiment.add_image(self.log_img_key, imgs, global_step = self.global_step)
    
    def configure_optimizers(self) -> None:
        optim_conf = self.conf['OPTIMIZER']

        # Initialize the optimizer
        optim = AdamW(
            self.parameters(), 
            **optim_conf,   
        )

        return optim

    # * ------ Samplers ------
    # * Deterministic solvers
    def heun(self):
        raise NotImplementedError()

    def dpmpp(
        self,
        shape : Tuple[int,...],
        schedule : Tensor,
        scaling  : Tensor,
        ctrl  : Optional[Tensor] = None,
        use_x_c : Optional[bool] = None,
        clamp : bool = False,
        guide : float = 1.,
        verbose : bool = False,
    ) -> Tensor:
        '''
            DPM++ Solver (2° order - 2M variant) from:
            https://arxiv.org/pdf/2211.01095 (Algorithm 2)
        '''

        use_x_c = default(use_x_c, self.self_cond)

        N = len(schedule)

        x_c = None # Parameter for self-conditioning
        x_t = schedule[0] * torch.randn(shape, device = self.device)

        logsnr = lambda sig : -log(sig)

        # Iterate through the schedule|scaling three at a time
        pars = zip(groupwise(schedule, n = 3, pad = -1, extend = False), scaling)
        for (sigm1, sig, sigp1), s in tqdm(pars, total = N, desc = 'DPM++', disable = not verbose):
            p_t = self.follow(x_t, sig, x_c = x_c if use_x_c else None, ctrl = ctrl, guide = guide, clamp = clamp)

            l_t, l_tp1 = logsnr(sig), logsnr(sigp1)
            h_tp1 : float = l_tp1 - l_t

            if x_c is None or sigp1 == 0:
                dxdt = p_t
            else:
                h_t = l_t - logsnr(sigm1)
                r_t = h_t / h_tp1

                dxdt = (1 + 1 / (2 * r_t)) * p_t - (1 / (2 * r_t)) * x_c
            
            x_t = sigp1 / sig * x_t - s * expm1(-h_tp1) * dxdt
            x_c = p_t

        return x_t

    def ddim(self):
        raise NotImplementedError()

    # * Stochastic Solvers
    def heun_sde(
        self,
        shape : Tuple[int,...],
        schedule : Tensor,
        scaling  : Tensor,
        ctrl  : Optional[Tensor] = None,
        use_x_c : Optional[bool] = None,
        clamp : bool = False,
        guide : float = 1.,
        s_tmin : float = 0.05,
        s_tmax : float = 50.,
        s_churn : float = 80,
        s_noise : float = 1.003,
        verbose : bool = False
    ) -> Tensor:
        '''
            Stochastic Heun (2° order) solver from:
            https://arxiv.org/pdf/2206.00364 (Algorithm 2)
        '''
        sqrt2m1 = 0.4142135624
        use_x_c = default(use_x_c, self.self_cond)

        T = len(schedule)

        # Compute the gamma coefficients that increase the noise level
        gammas = torch.where(
            (schedule < s_tmin) | (schedule > s_tmax),
            0., min(s_churn / T, sqrt2m1)
        )

        x_c = None # Parameter for self-conditioning
        x_t = schedule[0] * torch.randn(shape, device = self.device)

        pars = zip(groupwise(schedule, n = 2), gammas)
        for (sig, sigp1), gamma in tqdm(pars, total = T, desc = 'Stochastic Heun', disable = not verbose):
            # Sample additive noise
            eps = s_noise * torch.randn_like(x_t)

            # Select temporarily increased noise level sig_hat
            sig_hat = sig * (1 + gamma)

            # Add new noise to move schedule to time "hat"
            x_hat = x_t + sqrt(sig_hat ** 2 - sig ** 2) * eps

            # Evaluate dx/dt at scheduled time "hat"
            p_hat = self.follow(x_hat, sig_hat, x_c = x_c if use_x_c else None, ctrl = ctrl, guide = guide, clamp = clamp)
            dx_dt = (x_hat - p_hat) / sig_hat

            # Take Euler step from schedule time "hat" to time + 1
            x_t = x_hat + (sigp1 - sig_hat) * dx_dt

            # Add second order correction only if schedule not finished yet
            if sigp1 != 0:
                # Now the conditioning can be last prediction
                p_hat = self.follow(x_t, sigp1, x_c = x_c if use_x_c else None, ctrl = ctrl, guide = guide, clamp = clamp)
                dxdtp = (x_t - p_hat) / sigp1

                x_t = x_hat + 0.5 * (sigp1 - sig_hat) * (dx_dt + dxdtp)
            
            x_c = p_hat

        return x_t.clamp(-1., 1.) if clamp else x_t

    def dpmpp_sde(self):
        raise NotImplementedError()
