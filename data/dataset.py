from torch.utils.data.dataset import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len(self):
        return len(self.X)

    def __getitem_(self, idx):
        return self.X[idx], self.y[idx]