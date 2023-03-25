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
    
def coco_lbl_to_idx(lbls : Union[int, List[int], Tensor]) -> int:
    lbl_to_idx = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
                 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,18: 17, 19: 18,
                 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26,
                 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34,
                 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42,
                 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50,
                 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58,
                 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66,
                 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74,
                 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80, 92: 81, 93: 82,
                 95: 83, 100: 84, 107: 85, 109: 86, 112: 87, 118: 88, 119: 89,
                 122: 90, 125: 91, 128: 92, 130: 93, 133: 94, 138: 95, 141: 96,
                 144: 97, 145: 98, 147: 99, 148: 100, 149: 101, 151: 102, 154: 103,
                 155: 104, 156: 105, 159: 106, 161: 107, 166: 108, 168: 109, 171: 110,
                 175: 111, 176: 112, 177: 113, 178: 114, 180: 115, 181: 116, 184: 117,
                 185: 118, 186: 119, 187: 120, 188: 121, 189: 122, 190: 123, 191: 124,
                 192: 125, 193: 126, 194: 127, 195: 128, 196: 129, 197: 130, 198: 131,
                 199: 132, 200: 133}
    
    if isinstance(lbls, int): out = lbl_to_idx[int(lbls)]
    else: out = [lbl_to_idx[int(lbl)] for lbl in lbls]

    return out

def coco_idx_to_lbl(idxs : Union[int, List[int], Tensor]) -> int:
    idx_to_lbl = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
                 11: 11, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19,
                 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28,
                 27: 31, 28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38,
                 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46, 42: 47,
                 43: 48, 44: 49, 45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55,
                 51: 56, 52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62, 58: 63,
                 59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75,
                 67: 76, 68: 77, 69: 78, 70: 79, 71: 80, 72: 81, 73: 82, 74: 84,
                 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90, 81: 92, 82: 93,
                 83: 95, 84: 100, 85: 107, 86: 109, 87: 112, 88: 118, 89: 119,
                 90: 122, 91: 125, 92: 128, 93: 130, 94: 133, 95: 138, 96: 141,
                 97: 144, 98: 145, 99: 147, 100: 148, 101: 149, 102: 151, 103: 154,
                 104: 155, 105: 156, 106: 159, 107: 161, 108: 166, 109: 168, 110: 171,
                 111: 175, 112: 176, 113: 177, 114: 178, 115: 180, 116: 181, 117: 184,
                 118: 185, 119: 186, 120: 187, 121: 188, 122: 189, 123: 190, 124: 191,
                 125: 192, 126: 193, 127: 194, 128: 195, 129: 196, 130: 197, 131: 198, 
                 132: 199, 133: 200}
    
    if isinstance(idxs, int): out = idx_to_lbl[int(idxs)]
    else: out = [idx_to_lbl[int(idx)] for idx in idxs]

    return out