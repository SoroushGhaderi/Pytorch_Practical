import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
input_dim = 784  # 28*28
hidden_dim = 100
n_classes = 10
n_epoch = 2
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                           transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                          transform=transforms.ToTensor(), download=False)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet(input_size=input_dim, hidden_size=hidden_dim,
                  num_classes=n_classes)
