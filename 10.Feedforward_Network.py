import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_step = len(train_dataloader)
for epoch in range(n_epoch):
    for i, (images, labels) in enumerate(train_dataloader):
        # flat picture
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{n_epoch}], Step [{i+1}/{n_total_step}], Loss: {loss.item():.4f}')


# test and evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_dataloader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
