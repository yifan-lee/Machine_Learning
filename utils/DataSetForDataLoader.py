import torch
from torch.utils.data import Dataset


class numericDataset(Dataset):
    def __init__(self, X, y, Xclass, Yclass):
        self.X = torch.tensor(X, dtype=Xclass)
        self.y = torch.tensor(y, dtype=Yclass)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]