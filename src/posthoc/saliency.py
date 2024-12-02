import torch
import torch.nn as nn


class Saliency(nn.Module):
    def __init__(self, fn: callable):
        super().__init__()
        self.f = fn

    def forward(self, x):
        # the saliency is the gradient of the function with respect to the input
        x = torch.func.vmap( torch.func.jacrev(self.f))( x ) 

        # x dim is batch_size, m , batch_size, n
        return x


class InputXGradient(nn.Module):
    def __init__(self, fn: callable):
        super().__init__()
        self.f = fn

    def forward(self, x):
        grads = torch.func.vmap( torch.func.jacrev(self.f))( x ) 
        return torch.einsum("bmn, bn -> bmn", grads, x)
