import yaml
from abc import abstractmethod

from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from typing import List, Union, Callable, Optional

from .misc import default

class AbstractDM(LightningDataModule):
    '''
        Abstract Lightning Data Module that represents a dataset we
        can train a Lightning module on.
    '''

    def __init__(
        self,
        batch_size : int = 16,
        num_workers : int = 0,
        train_shuffle : bool = True,
        val_shuffle   : bool = False,
        val_batch_size : Optional[int] = None,
        worker_init_fn : Optional[Callable] = None,
        collate_fn     : Optional[Callable] = None,
        train_sampler  : Optional[Callable] = None, 
        val_sampler    : Optional[Callable] = None,
        test_sampler   : Optional[Callable] = None, 
    ) -> None:
        super().__init__()

        self.train_dataset = None
        self.valid_dataset = None
        self.test__dataset = None

        val_batch_size = default(val_batch_size, batch_size)

        self.num_workers    = num_workers
        self.batch_size     = batch_size
        self.train_shuffle  = train_shuffle
        self.val_shuffle    = val_shuffle
        self.train_sampler  = train_sampler
        self.valid_sampler  = val_sampler
        self.test__sampler  = test_sampler
        self.collate_fn     = collate_fn
        self.worker_init_fn = worker_init_fn
        self.val_batch_size = val_batch_size

    @classmethod
    def from_conf(cls, conf_path : str, key : str = 'DATASET') -> 'AbstractDM':
        '''
            Construct a Lightning DataModule from a configuration file.
        '''

        with open(conf_path, 'r') as f:
            conf = yaml.safe_load(f)

        data_conf = conf[key]

        return cls(
            **data_conf,
        )

    @abstractmethod
    def setup(self, stage: str) -> None:
        msg = 'This is an abstract datamodule class. You should use one of ' +\
              'the concrete subclasses that represents an actual dataset.'

        raise NotImplementedError(msg)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            sampler        = self.train_sampler,
            batch_size     = self.batch_size,
            shuffle        = self.train_shuffle,
            collate_fn     = self.collate_fn,
            num_workers    = self.num_workers,
            worker_init_fn = self.worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            sampler        = self.valid_sampler,
            batch_size     = self.val_batch_size,
            shuffle        = self.val_shuffle,
            collate_fn     = self.collate_fn,
            num_workers    = self.num_workers,
            worker_init_fn = self.worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test__dataset,
            sampler        = self.test__sampler,
            batch_size     = self.val_batch_size,
            shuffle        = self.val_shuffle,
            collate_fn     = self.collate_fn,
            num_workers    = self.num_workers,
            worker_init_fn = self.worker_init_fn,
        )