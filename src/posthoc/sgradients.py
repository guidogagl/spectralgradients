import torch
import torch.nn as nn
from torch.autograd import Function

from matplotlib import pyplot as plt

import numpy as np
from scipy import signal

# import tqdm based if used in notebook or not
from tqdm.autonotebook import tqdm

from src.posthoc.integrals import fline_integral


class SpectralGradients(nn.Module):
    def __init__(
        self,
        fn: callable,
        fs: float,
        Q: int,
        nperseg: int,
        noverlap: int = None,
        strategy: str = "highest",
        n_points: int = 2,
        line_integral: callable = fline_integral,
    ):
        super().__init__()
        if noverlap is None:
            noverlap = nperseg // 2

        # Compute the frequency vector of the power spectral density using the welch method
        self.fs = fs
        self.nperserg = nperseg
        self.noverlap = noverlap

        freqs, _ = signal.welch(
            torch.zeros(nperseg), fs, nperseg=nperseg, noverlap=noverlap
        )

        self.f = torch.tensor(freqs, dtype=torch.float64)
        self.f[0] = self.f[1]
        self.f[-1] = self.f[-2]

        # construct the bands vector with adjacent frequencies
        # self.f[i] -> [f_i, f_{i+1}], remove the last band as it overlaps with the previous one
        self.f = torch.cat(
            [self.f[:-1].unsqueeze(0), self.f[1:].unsqueeze(0)], dim=0
        ).transpose(0, 1)

        # Compute the order of each frequency band based on the strategy:
        # - "highest": order the bands based on the highest frequency
        # - "lowest": order the bands based on the lowest frequency
        # TODO: implement the other strategies "more_power" and "less_power"

        self.filters = nn.ModuleList()
        self.filters.append(nn.Identity())
        self.filters.append(nn.Identity())

        if strategy == "highest":
            self.freq_index = torch.argsort(self.f[:, 0], descending=True).long()

            high_freq = self.f[-1, 1].item()  # high frequency is fs/2 - 1
            self.f[:, 1] = high_freq

        elif strategy == "lowest":
            self.freq_index = torch.argsort(self.f[:, 0], descending=False).long()

            low_freq = self.f[0, 0].item()
            self.f[:, 0] = low_freq

        else:
            raise ValueError("strategy must be either 'highest' or 'lowest'")

        for index in self.freq_index[2:-1]:
            f_i, f_i1 = self.f[index]
            low_freq = f_i.item()
            self.filters.append(BandStopFilter(low_freq, high_freq, fs, Q))

        self.filters.append(SpectralErase())

        self.line_integral = line_integral
        self.fun = fn
        self.n_points = n_points

    def forward(self, x):
        # x shape is batch_size, n
        # the function will return batch_size, m, freqs, n

        gradients = []
        x_i = x

        for filter in self.filters:

            x_i1 = filter(x)

            grads = self.line_integral(self.fun, x_i1, x_i, self.n_points).cpu()

            gradients += [grads]

            x_i = x_i1

        gradients = torch.stack(gradients, dim=2)

        gradients = gradients[:, :, self.freq_index]

        return gradients.to(x.device)

    def plot_power_spectral_density(self, x):
        # x shape is (1, n)
        fs, nperseg, noverlap = self.fs, self.nperserg, self.noverlap

        x = x.view(1, -1)

        for i, filter in enumerate(self.filters):
            if i % 20 == 0:
                filt_x = filter(x)
                f, Pxx = signal.welch(
                    filt_x.view(-1).numpy(), fs, nperseg=nperseg, noverlap=noverlap
                )
                plt.plot(f, Pxx, label=f"filter_{i}")

        filter = self.filters[-1]

        filt_x = filter(x)
        f, Pxx = signal.welch(
            filt_x.view(-1).numpy(), fs, nperseg=nperseg, noverlap=noverlap
        )
        plt.plot(f, Pxx, label=f"filter_last")

        plt.legend()
        plt.show()


class SpectralErase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0


class BandStopFilterFun(Function):
    @staticmethod
    def forward(ctx, input, a, b):
        # detach so we can cast to NumPy
        input, a, b = input.detach(), a.detach(), b.detach()

        in_device = input.device

        ctx.save_for_backward(input, a, b)

        result = signal.filtfilt(
            b.cpu().numpy(), a.cpu().numpy(), input.cpu().numpy().astype(np.float64)
        ).copy()
        return torch.as_tensor(result, dtype=input.dtype, device=in_device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, a, b = ctx.saved_tensors

        return input, a, b


class BandStopFilter(nn.Module):
    def __init__(self, low_freq, high_freq, sample_rate, Q):
        super().__init__()

        b, a = signal.butter(Q, [low_freq, high_freq], btype="bandstop", fs=sample_rate)

        # Register the filter coefficients as buffers
        self.register_buffer("b", torch.tensor(b, dtype=torch.float64))
        self.register_buffer("a", torch.tensor(a, dtype=torch.float64))

    def forward(self, x):
        return BandStopFilterFun.apply(x, self.a, self.b)
