import torch

a = torch.randn(1, 3, 10)
print(a)

b = a[:, -1, :]
print(b)
print(b.shape)