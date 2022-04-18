import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, data_dir):
        self.X, self.y = np.load(data_dir)
        
    def __len(self):
        return len(self.X)

    def __getitem_(self, idx):
        return self.X[idx], self.y[idx]