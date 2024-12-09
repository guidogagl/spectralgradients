import torch
import torch.nn as nn

from src.posthoc.integrals import step_integral

from typing import Union

from torchaudio.functional import lowpass_biquad, highpass_biquad


def stopband_filter(x, fs, cutoff, filter_type="low", order=5):

    if filter_type == "low":
        x = lowpass_biquad( x, sample_rate = fs, cutoff = cutoff, Q = order)
    elif filter_type == "high":
        x = highpass_biquad( x, sample_rate = fs, cutoff = cutoff, Q = order)
    else:
        raise ValueError("filter_type must be 'low' or 'high'")
    return x


def freqs_path_integral(fn: nn.Module, x, fs, nperseg, Q=5, strategy="highest"):
    nyquist = 0.5 * fs
    freq_res = fs / nperseg

    SG, W = [], []

    steps = torch.arange(0, nyquist + freq_res, freq_res)

    if strategy == "highest":  # remove the high frequency components first
        steps = steps.flip( dims = 0)
        filter_type = "low"
    elif strategy == "lowest":
        filter_type = "high"

    steps = steps[1:]
    start = x

    for i, step in enumerate(steps):
    
        if step == 0 or step == nyquist:
            end = torch.zeros_like(x)
        else:
            end = stopband_filter(x, fs, step, filter_type, Q).to(x.device)

        def path(alpha):
            return start * alpha - (1 - alpha) * end.to(x.device)

        def jac_path(alpha):
            return start - end

        W = W + [ torch.abs(nn(start) - nn(end)).cpu()]
        SG = SG + [step_integral(fn, step, i, path, jac_path, x.device).cpu()]

        start = end

    W = torch.stack( W, dim = 0).permute(1, 2, 0) # b, m , f
    SG = torch.stack(SG, dim = 0).permute(1, 2, 0, 3) # b, m , f,  n

    return W, SG


class SpectralGradients(nn.Module):
    def __init__(
        self,
        f: nn.Module,
        fs: float,
        Q: int,
        nperseg: int,
        strategy: str = "highest",
    ):
        super().__init__()

        self.fn = f
        self.Q = Q
        self.nperseg = nperseg
        self.fs = fs
        self.strategy = strategy

    def forward(self, x):
        return freqs_path_integral(
            fn=self.fn,
            x=x,
            fs=self.fs,
            nperseg=self.nperseg,
            Q=self.Q,
            strategy=self.strategy,
        )