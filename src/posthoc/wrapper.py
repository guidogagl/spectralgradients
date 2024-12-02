import torch
from torch import nn

from typing import Tuple


class ExWrapper(nn.Module):
    def __init__(
        self,
        func: callable,
        in_shape: Tuple[int],
        target: Tuple[int] = None,
        softmax: bool = True,
    ):

        self.in_shape = in_shape
        self.target = target

        if softmax:
            func = nn.Sequential(
                func,
                nn.Softmax(dim=-1),
            )

        self.model = func

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.reshape(batch_size, *self.in_shape)

        x = self.model(x)

        if self.target is not None:
            x = x[..., *self.target]

        x = x.reshape(batch_size, -1)

        return x
