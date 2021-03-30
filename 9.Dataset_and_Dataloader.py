import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self):
        # load data
        data = np.loadtxt("./data/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(data[:, 1:])
        self.y = torch.from_numpy(data[:, 0])
        self.n_samples = data.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(first_data)
print(features, labels)