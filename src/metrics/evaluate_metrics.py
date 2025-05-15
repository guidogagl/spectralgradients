import sys, os
sys.path.append("./")

import torch
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
 
from src.metrics.evaluate_tf import Localization, Complexity, Infidelity, convert_to_tf, Wrapper
from src.synt_data import SyntDataset, SETUPS
from src.train.model import TimeModule
from src.posthoc.sgradients import SpectralGradients

from torch.utils.data import SubsetRandomSampler, DataLoader


setups = [0]
strategies = ["highest", "lowest"]
metrics = ["comp", "inf", "loc"]

batch_size = 256
nperseg = 100
fs = 100

n_points = 10


def plot_spectralgradients(
    signal,
    SG,
    W,
    mask,
    output_dir
):
    t = np.linspace( 0, 10, 1000 )
    f = np.linspace( 0, 50, 50 )
    
    fig, axs = plt.subplots(3, 2, figsize=(10, 10), width_ratios=(1, 5) )
    fig.delaxes(axs[ 0, 0] )
     
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].set_xticks([])
    axs[0, 1].spines['top'].set_visible(False)
    axs[0, 1].spines['right'].set_visible(False)  

    axs[0, 1].plot(
        t,
        signal.reshape( -1 ),
        linewidth=0.5,
        color="black",
        alpha=0.5,
    )
    
    axs[1, 1].pcolormesh(
        t,
        f,
        mask,
        shading="gouraud",
        cmap="viridis",
        #extent=extent,
    )
    
    axs[1, 1].set_ylabel("Frequency (Hz)")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_xticks([])    
    axs[1, 1].spines['top'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)
    
    
    axs[2, 1].pcolormesh(
        t,
        f,
        SG,
        shading="gouraud",
        cmap="viridis",
        #extent=extent,
    )
    
    axs[2, 1].set_ylabel("Frequency (Hz)")
    axs[2, 1].set_xlabel("Time (s)")

    axs[2, 1].spines['top'].set_visible(False)
    axs[2, 1].spines['right'].set_visible(False)
    
    
    axs[2, 0].set_title( f"W")
    axs[2, 0].set_ylabel("Frequency (Hz)")
    axs[2, 0].set_xlabel("Importance")
    # plot with a thin line
        
    axs[2, 0].plot(
        W,
        f,
        linewidth=0.5,
        color="red",
        alpha=0.5,
    )
    
    axs[2, 0].spines['top'].set_visible(False)
    axs[2, 0].spines['right'].set_visible(False)
    
    
    W = 1 - W
    W[::-1].sort()    
    axs[1, 0].set_title( f"Infidelity Curve")
    axs[1, 0].set_xlabel("Frequency (Hz)")
    axs[1, 0].set_ylabel("Infidelity")
    # plot with a thin line
        
    axs[1, 0].plot(
        f,
        W,
        linewidth=0.5,
        color="red",
        alpha=0.5,
    )
    
    axs[1, 0].spines['top'].set_visible(False)
    axs[1, 0].spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(
        output_dir,
        f"spectral_gradients.png"
    )    
    
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    
    return        

comp = Complexity()
loc = Localization()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for setup in setups:
    
    data = SyntDataset(setup = setup, nperseg=nperseg, return_mask=True)

    loader = DataLoader(
        data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(list(range(len(data) - data.n_samples))),
        num_workers=7,
    )

    nn = TimeModule.load_from_checkpoint(
        f"output/model/setup{setup}/checkpoint.ckpt",
        input_shape=data[0][0].shape,
        fs=100,
        n_classes=data.n_class,
    ).nn.eval()
    
    nn = Wrapper( nn, data[0][0].shape ).to(device)
    
    inf = Infidelity( f = nn ).to(device)
    
    for strategy in strategies:
        
        checkpoint_path = f"output/model/setup{setup}/metrics/sg/{strategy}/"
    
        explainer = SpectralGradients(
            f = nn,
            fs = fs,
            Q = 5,
            nperseg = nperseg,
            strategy = strategy,
            n_points = n_points,
        ).to(device)
        
        for x, mask, y in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            attr = explainer( x )

            attr = ( attr[0].to(device), attr[1].to(device) )
            mask = convert_to_tf( mask, y, setup)
            
            l = loc( x, attr, mask ).detach().cpu().numpy()
            c = comp( x, attr, mask ).detach().cpu().numpy()
            i = inf( x, attr, mask ).detach().cpu().numpy()
            
            W, SG = attr
            SG = torch.einsum( "bcf,bcft->bcft", W.to(device), SG.to(device) ).detach().cpu().numpy()
            W = W.detach().cpu().numpy()
            
            y = y.detach().cpu().numpy()
            
            # reduce to the true classes
            l = np.array( [ l[j, y[j]] for j in range(len(l)) ] )
            c = np.array( [ c[j, y[j]] for j in range(len(c)) ] )
            i = np.array( [ i[j, y[j]] for j in range(len(i)) ] )
           
            print( "Setup: ", setup )
            print( "Localization: ", l.mean() )
            print( "Complexity: ", c.mean() )
            print( "Infidelity: ", i.mean() )

            plot_spectralgradients(
                x[0].cpu().numpy(),
                SG[0][y[0]],
                W[0][y[0]],
                mask[0].cpu().numpy(),
                output_dir = checkpoint_path,
            )
            
            del SG, W, x, y, mask, attr
            torch.cuda.empty_cache()
            
            break
            exit()