import torch
import numpy as np

# Create empty 2D tensor
x_2d = torch.empty(3, 2)
print(x_2d)

# Create empty 3D tensor
x_3d = torch.empty(2, 4, 2)
print(x_3d)

# Create random 2D tensor
x_random = torch.rand(2, 2, dtype=torch.float)
print(x_random)

# Check size of x_random tensor
print("x_random size: {}".format(x_random.size()))

# Create 1D tensor
x_tensor = torch.tensor([2, 1, 1.2, 3])
print(x_tensor)

# Addition
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
z = x + y
print(z)

# In-place addition
y.add_(x)
print(y)

# Print value with tensor notation
print(x[1, 1])

# Print only item of tensor
print(x[1, 1].item())

# Convert torch tensors to numpy arrays
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# Check cuda availability
if torch.cuda.is_available():
    device = torch.device("cuda")

# print device
print(device)

# Convert numpy array to tensor
a = np.ones(3)
print(a)
b = torch.from_numpy(a)
print(b)

# flag the requires_grad= True
requires_grad = torch.ones(3, requires_grad=True)
print(requires_grad)