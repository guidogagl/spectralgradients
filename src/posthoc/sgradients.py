import torch
import torch.nn as nn

from src.posthoc.integrals import step_integral, fline_integral

from typing import Union

from scipy import signal

def stopband_filter(x, fs, cutoff, filter_type="low", order=5):
    
    if filter_type == "low":
        b, a = signal.butter(order, cutoff, fs = fs, btype='lowpass')
    elif filter_type == "high":
        b, a = signal.butter(order, cutoff, fs = fs, btype='highpass')
    else:
        raise ValueError("filter_type must be 'low' or 'high'")

    y = signal.filtfilt(b, a, x.clone().detach().cpu().numpy()).copy()

    return torch.tensor( y, dtype = x.dtype, device=x.device)


def freqs_path_integral(fn: nn.Module, x, fs, nperseg, Q=5, strategy="highest", n_points = 50):
    nyquist = 0.5 * fs
    freq_res = fs / nperseg

    SG, W = [], []

    steps = torch.arange(0, nyquist + freq_res, freq_res)

    if strategy == "highest":  # remove the high frequency components first
        steps = steps.flip(0)
        filter_type = "low"
    elif strategy == "lowest":
        filter_type = "high"

    #steps = steps[1:]
    start = x

    # Den = 1 / ( fn( x ) - fn ( torch.zeros_like(x) )) 

    for i in range(1, len(steps)):
        
        step = steps[i]

        if step == 0 or step == nyquist:
            end = torch.zeros_like(x)
        else:
            end = stopband_filter(x, fs, step, filter_type, Q)

        def path(alpha):
            return start * alpha - (1 - alpha) * end

        def jac_path(alpha):
            return start - end

        W = W + [ fn(start) - fn(end) ]
        #SG = SG + [step_integral(fn, steps, i, path, jac_path, x.device)]
        SG = SG + [fline_integral(fn, start, end, n_points)]
        start = end

    W = torch.stack( W, dim = 0).permute(1, 2, 0) # b, m , f
    SG = torch.stack(SG, dim = 0).permute(1, 2, 0, 3) # b, m , f,  n

    if strategy == "highest":
        W = W.flip(2)
        SG = SG.flip(2)

    # W = torch.einsum( "bmf,bm->bmf", W, Den)
    W = torch.nn.functional.relu( W )
    W = torch.round(W, decimals = 3)

    W = torch.einsum( "bmf,bm->bmf", W, 1 / (W.sum(dim = -1) + 1e-8))   
    return W, SG


class SpectralGradients(nn.Module):
    def __init__(
        self,
        f: nn.Module,
        fs: float,
        Q: int,
        nperseg: int,
        strategy: str = "highest",
        n_points : int = 2,
    ):
        super().__init__()

        self.fn = f
        self.Q = Q
        self.nperseg = nperseg
        self.fs = fs
        self.strategy = strategy
        self.n_points = n_points
    def forward(self, x):
        return freqs_path_integral(
            fn=self.fn,
            x=x,
            fs=self.fs,
            nperseg=self.nperseg,
            Q=self.Q,
            strategy=self.strategy,
            n_points=self.n_points,
        )
    

def plot_sg(
    SG : nn.Module,
    signal : torch.Tensor,
    label : int,
    fs : float = 100,
    nperseg : float = 200,
    save_path : str = None
    ):

    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import scipy.signal as Fun

    sns.set_theme(
        #context='notebook', 
        style='darkgrid', 
        palette='deep', 
        font='sans-serif', 
        font_scale=1.2, 
        color_codes=True, 
        rc={'figure.figsize':(5, 2.5)}
    ) 

    df = pd.DataFrame([])
    df["Amplitude"] = signal.view(-1).cpu().numpy()
    df["Time (s)"] = torch.arange(len(signal)) / fs
    sns.lineplot(df, x = "Time (s)", y = "Amplitude")
    plt.title( "Input Signal")
    if save_path is not None:
        plt.savefig( f"{save_path}/input.png", dpi = 300)
    plt.show()

    pad_signal = torch.cat( [signal[:fs], signal] ).view( -1 ).cpu() 
    F, T, Sxx = Fun.spectrogram(
        pad_signal.view(-1).numpy(), fs=fs, noverlap = fs-1, nperseg=2*fs, window = "hann",
    )
    plt.pcolormesh(T, F, Sxx, shading='gouraud', cmap="coolwarm")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title( "Spectrogram")

    if save_path is not None:
        plt.savefig( f"{save_path}/spectrogram.png", dpi = 300)

    plt.show()

    W, S = SG( signal.unsqueeze(0) )

    W = W[0, 1].view(-1).detach().cpu().unsqueeze(1).numpy()
    
    S = S.detach().squeeze(0)[1]    
    S = S.reshape( S.size(0), S.size(1) // fs, -1).detach().cpu().numpy()
    S = S.sum( axis = -1)
    #S = torch.tensor(S)
    #S = torch.nn.functional.relu(S).numpy()    
    
    F = np.arange( 0, fs / 2, fs / nperseg )
    T = np.arange( 0, len(signal) // fs )

    fig, axes = plt.subplots( nrows=1, ncols=2, width_ratios = [0.05, 0.95] )
    axes = axes.flatten()
    sns.heatmap(np.flip(W, axis=0), cmap="Blues", ax= axes[0], cbar = False)
    axes[0].set_title( "W" )
    axes[0].set_ylabel('Frequency [Hz]')

    axes[0].set_xticks( ticks = [], labels = [])
    axes[0].set_yticks( ticks = [], labels = [])

    axes[1].pcolormesh(T, F, S, shading='gouraud', cmap="Blues")
    axes[1].set_yticks( ticks = [], labels = [])
    axes[1].set_xlabel('Time [sec]')
    axes[1].set_title( "SG" )

    if save_path is not None:
        plt.savefig( f"{save_path}/BI_{SG.strategy}.png", dpi = 300)
    plt.show()
    
    S = torch.tensor(S)
    W = torch.tensor(W).reshape(-1)
    S = torch.einsum( "f,fn->fn", W.cpu(), S.cpu()).detach().cpu().numpy()
    
    S = ( S - S.min())/(S.max() - S.min())

    F = np.arange( 0, fs / 2, fs / nperseg )

    plt.pcolormesh(T, F, S, shading='gouraud', cmap="Blues")
    #plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    if SG.strategy == "highest":
        plt.title( "SG path: highest to lowest")
    else:
        plt.title( "SG path: lowest to highest")

    if save_path is not None:
        plt.savefig( f"{save_path}/SG_{SG.strategy}.png", dpi = 300)
    plt.show()
