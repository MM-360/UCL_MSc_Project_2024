
import torch

from torch.utils.data import DataLoader, TensorDataset, Dataset


class SlideWindowDataset(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        return x

    def __len__(self):
        return len(self.data) - self.window