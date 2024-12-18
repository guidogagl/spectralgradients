import torch
import torch.nn as nn

from src.posthoc.integrals import step_integral

from typing import Union

from src.posthoc.sgradients import freqs_path_integral


class TimeSpectralGradients(nn.Module):
    def __init__(
        self,
        f: nn.Module,
        fs: float = 100,
        nperseg: int = 200,
        Q: int = 5,
        strategy: str = "highest",
        n_points : int = 2
    ):
        super().__init__()

        self.fn = f
        self.Q = Q
        self.nperseg = nperseg
        self.fs = fs
        self.strategy = strategy
        self.n_points = n_points

    def forward(self, x):
        W, SG = freqs_path_integral(
            fn=self.fn,
            x=x,
            fs=self.fs,
            nperseg=self.nperseg,
            Q=self.Q,
            strategy=self.strategy,
            n_points=self.n_points
        )

        return torch.einsum( "bmf,bmfn->bmn", W, SG)
