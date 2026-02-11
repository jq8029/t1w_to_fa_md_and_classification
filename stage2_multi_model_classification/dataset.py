
import torch
import numpy as np
from torch.utils.data import Dataset

class Dataset(torch.utils.data.Dataset):
    """
      X: [B, C, D, H]
      Y: [B, 0/1]
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, augment: bool = True):
        unique_labels = np.unique(Y)
        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self):
        return self.X.shape[0]

    def _random_flip(self, x):
        # x: (C, D, H)
        if np.random.rand() < 0.5:  # flip D
            x = torch.flip(x, dims=[1])
        if np.random.rand() < 0.5:  # flip H
            x = torch.flip(x, dims=[2])  
        return x

    def _intensity_jitter(self, x, scale=0.1):
        factor = 1.0 + (torch.rand(1) - 0.5) * 2 * scale
        x = x * factor
        return x

    def __getitem__(self, idx):
        x = self.X[idx]  # [B,C,D,H]
        y = self.Y[idx]  # [B, 0/1]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)   
        
        if self.augment:
            x = self._random_flip(x)
            x = self._intensity_jitter(x, scale=0.03)

        return x, y
