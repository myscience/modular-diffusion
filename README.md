# Modular Diffusion

PyTorch (Lightning) implementation of several diffusion models using the modular design as introduced in [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/pdf/2206.00364.pdf), (Karras et al. NIPS 2022). 

## Usage

Example training script the `ElucidatedDiffusion` model from Karras et al. (2022).

```python
from lightning import Trainer

from diffusion.module.utils.cifar10 import CIFAR10DM
from diffusion.elucidated import ElucidatedDiffusion

 # Load the model configuration file
conf_file = ... # path to YAML configuration file

# Initialize model and dataset from configuration file
model = ElucidatedDiffusion.from_conf(conf_file)
cifar = CIFAR10DM.from_conf(conf_file)

# Lightning Trainer for flexible accelerated training
trainer = Trainer(
    max_epochs : 500,
    accelerator = 'gpu',
    devices = 4, # Piece of cake multi-gpu support!
    strategy : 'ddp_find_unused_parameters_false',

)

trainer.fit(model, datamodule = cifar)
```

Example of the flexible `sampler` choice at inference time.

```python
from diffusion.elucidated import ElucidatedDiffusion

model = ElucidatedDiffusion.load_from_checkpoint(PATH)

# Sample using the 2° order stochastic
# Heun method from Kallas et al.
heun_imgs = model(
    num_imgs = 8,
    num_steps = 25, # Fast solver!
    ode_solver = 'heun_sde',
)

# Sample using the DPM++ Solver from Lu et al. (2022)
# (https://arxiv.org/pdf/2211.01095.pdf)
dpm_imgs = model(
    num_imgs = 8,
    num_steps = 25, # Fast solver!
    ode_solver = 'dpm++',
)

# Save or visualize the images
```

Example of a diffusion model with `class-conditioning` control.

```python

from diffusion.elucidated import ElucidatedDiffusion
from diffusion.module.components.embedding import ClassEmbedder

model = ElucidatedDiffusion.load_from_checkpoint(PATH)

# Set the control-embedder of the model to enable
# training with external conditioning
model.ctrl_emb = ClassEmbedder(
    emb_dim = 32, # Embedding dimension
    cls_dim = 10, # Number of classes
    p_dropb = .1, # Dropout probability for conditioning
)

# Train the model with Lightning Trainer 
```

## References

The code is *heavily* based on the beautiful (diffusion) repositories by [lucidrains](https://github.com/lucidrains/denoising-diffusion-pytorch) and [crowsonkb](https://github.com/crowsonkb/k-diffusion/tree/b43db16749d51055f813255eea2fdf1def801919).

```bibtex
@article{karras2022elucidating,
  title={Elucidating the design space of diffusion-based generative models},
  author={Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  journal={arXiv preprint arXiv:2206.00364},
  year={2022}
}
```

```bibtex
@misc{chen2022analog,
    title   = {Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning},
    author  = {Ting Chen and Ruixiang Zhang and Geoffrey Hinton},
    year    = {2022},
    eprint  = {2208.04202},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```