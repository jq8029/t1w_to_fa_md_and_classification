import torch
import numpy as np
from torch.utils.data import Dataset

class Dataset(torch.utils.data.Dataset):
    """
      INPUT:  [B, C, D, H, W]
    """
    def __init__(self, INPUT: np.ndarray, OUTPUT: np.ndarray, augment: bool = False):
        self.INPUT = INPUT
        self.OUPUT = OUTPUT
        self.augment = augment
    
    def __len__(self):
        return self.INPUT.shape[0]

    def _random_flip_(self, input: np.ndarray, output: np.ndarray):
        # Random flips along each axis (p=0.5)
        # input:  [C,D,H,W]
        # output: [C,D,H,W]
        if np.random.rand() < 0.5: 
            input, output = input[:, ::-1, :, :].copy(), output[:, ::-1, :, :].copy() # flip depth
        if np.random.rand() < 0.5: 
            input, output = input[:, :, ::-1, :].copy(), output[:, :, ::-1, :].copy() # flip height
        if np.random.rand() < 0.5: 
            input, output = input[:, :, :, ::-1].copy(), output[:, :, :, ::-1].copy() # flip width

        return input, output

    def __getitem__(self, idx):
        input = self.INPUT[idx]   # [C,D,H,W]
        output = self.OUPUT[idx]  # [C,D,H,W]

        if self.augment:
            input, output = self._random_flip_(input, output)

        input  = torch.from_numpy(input.astype(np.float32))
        output = torch.from_numpy(output.astype(np.float32))

        return input, output

