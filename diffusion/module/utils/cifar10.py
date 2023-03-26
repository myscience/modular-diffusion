import torch
import torch.nn as nn

from numpy import array
from torch import Tensor

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from .data import AbstractDM

from PIL import Image

from typing import Union

def rgb2torch(pic : Union[Image.Image, Tensor]) -> Tensor:
    if isinstance(pic, Tensor): return pic

    w, h, c = *pic.size, len(pic.getbands())
    img = torch.as_tensor(array(pic, copy = True), dtype = torch.uint8)
    img = img.view(h, w, c).permute((2, 0, 1)).contiguous()

    return img

class condCIFAR10(Dataset):

    labels = ["airplane", "automobile",  "bird",  "cat",  "deer",
              "dog", "frog", "horse", "ship", "truck"]

    lbl_dict = {
        0 : 'airplane',
        1 : 'automobile',
        2 : 'bird',
        3 : 'cat',
        4 : 'deer',
        5 : 'dog',
        6 : 'frog',
        7 : 'horse',
        8 : 'ship',
        9 : 'truck'
    }

    def __init__(
        self,
        root : str,
        train : bool = True,
        transform = None,
    ) -> None:
        super().__init__()

        self.cifar = CIFAR10(root, train = train, transform = transform, download = True)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, index) -> dict:
        img, lbl = self.cifar[index]

        data = {
            # Get the image from CIFAR10
            'smap' : img,
            'lbls' : lbl, 
        }

        return data

    @classmethod
    def cond2lbl(cls, cond):
        return [cls.lbl_dict[int(c)] for c in cond.view(-1)]

class CIFAR10DM(AbstractDM):

    def __init__(
        self,
        root : str,
        token_dim : int = 4,
        seq_len   : int = 4,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs
        )

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Lambda(rgb2torch),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x : 2 * x - 1.)
        ])

        self.root = root
        self.seq_len = seq_len
        self.token_dim = token_dim
        self.transform = transform

    def setup(self, stage = None):
        cifar_train = condCIFAR10(self.root,
                        train = True,
                        transform = self.transform,
                        token_dim = self.token_dim,
                        seq_len = self.seq_len,
                    )

        cifar_val = condCIFAR10(self.root,
                        train = False,
                        transform = self.transform,
                        token_dim = self.token_dim,
                        seq_len = self.seq_len,
                    )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = cifar_train
            self.valid_dataset = cifar_val

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = cifar_val