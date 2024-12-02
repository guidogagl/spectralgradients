import torch
import torch.nn as nn

from src.posthoc.integrals import fline_integral

from typing import Union


class IntegratedGradients(nn.Module):
    def __init__(self,
        fn : nn.Module,
        baselines : torch.Tensor,
        n_points : int = 50,        
        ):
        
        super(IntegratedGradients, self).__init__()

        self.fun = fn
        self.register_buffer(  "baselines", baselines )
        self.n_points = n_points

    def forward(
        self,
        x: torch.Tensor
    ):
        return fline_integral(self.fun, x, self.baselines, self.n_points )

