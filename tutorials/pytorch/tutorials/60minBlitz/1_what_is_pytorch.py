import torch
import numpy as np

## Tensors ##

# ininitialized
x = torch.empty(5, 3)
print(x)

# randomly initialized
x = torch.rand(5, 3)
print(x)

# zero matrix
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# tensor from data
x = torch.tensor([5.5, 3])
print(x)

# reuse properties of tensors
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)

# get size
print(x.size())



## Operations ##

y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

# numpy-like indexing
print(x[:, 1])

# resizing: view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())


# one element tensor -> python number
x = torch.randn(1)
print(x)
print(x.item())



## NumPy Bridge ##

# torch tensor -> numpy array
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# numpy array -> torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


## CUDA tensors ##

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))


