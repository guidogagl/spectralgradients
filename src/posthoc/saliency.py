import torch
import torch.nn as nn


class Saliency(nn.Module):
    def __init__(self, f: nn.Module, name: str = "sal", **kwargs):
        super().__init__()
        self.f = f
        self.name = name

    def forward(self, x):
        # the saliency is the gradient of the function with respect to the input
        grads = torch.func.vmap(torch.func.jacrev(self.f))(x)
        # x dim is batch_size, m , batch_size, n
        return grads


class InputXGradient(nn.Module):
    def __init__(self, f: nn.Module, name: str = "ixg", **kwargs):
        super().__init__()
        self.f = f
        self.name = name

    def forward(self, x):
        grads = torch.func.vmap(torch.func.jacrev(self.f))(x)

        return torch.einsum("bmn, bn -> bmn", grads, x)
