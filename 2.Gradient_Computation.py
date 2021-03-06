import torch

# create random tensor and set requires grad equals True
x = torch.randn(3, requires_grad=True)
print(x)

# pytorch detect add operation
y = x + 2
print(y)

# pytorch detect mul operation
z = y * y * 2
z = z.mean()
print(z)

# dz/dx
z.backward()
print(x.grad)

# we can set requires_grad equals false in 3 ways
# x.requires_grad_(False)
# x.detach
# with torch.no_grad():

# first
print(x)
x.requires_grad_(False)
print(x)

# second
print(x)
x.detach()
print(x)

# third
with torch.no_grad():
    y = x + 2
    print(y)

