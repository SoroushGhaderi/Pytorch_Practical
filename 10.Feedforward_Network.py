import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
input_size = 784  # 28*28
hidden_size = 100
n_classes = 10
n_epoch = 2
batch_size = 100
learning_rate = 0.001

#MNIST
train_dataset = torchvision.datasets.MNIST(root=)

