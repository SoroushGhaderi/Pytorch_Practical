import torch
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
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

num_epoch = 2
total_sample = len(dataset)
n_iterations = math.ceil(total_sample/4)
print(total_sample, n_iterations)

for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, update
        if (i+1) % 5 == 0:
            print("epoch {}/{}, step {}/{}"
                  .format(epoch+1, num_epoch, i+1, n_iterations))
