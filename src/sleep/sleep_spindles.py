import os 
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import torch    
from torch import nn
import numpy as np
import pandas as pd

from lightning.pytorch import seed_everything
from torch import set_float32_matmul_precision

from physioex.train.models import load_model
from physioex.train.networks import config as net_config 
from physioex.data import PhysioExDataModule

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

import sys 
sys.path.append( os.getcwd() )

from src.sleep.utils import evaluate_model, get_importances, isolate_bands
from src.sleep.train import TinyTransformerNet
from src.posthoc.sgradients import SpectralGradients
from physioex.train.networks.tinysleepnet import TinySleepNet

from scipy.signal import spectrogram

batch_size = 256
dataset = "sleepedf"
net = "tinysleepnet"
data_folder = "/scratch/leuven/365/vsc36564/"
device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_bands(
    index  : int,
    signal : np.ndarray,
    preds  : np.ndarray,
):
    
    bands = isolate_bands(  signal )

    fig, axs = plt.subplots( 7, 1, figsize=(6, 9), sharex=True, sharey=True ) 
    
    bands_name = ["Raw", "Gamma", "Beta", "Sigma", "Alpha", "Theta", "Delta"]
    
    for i in range( 7 ):
        axs[i].set_title( bands_name[i] )
        axs[i].set_ylabel("Amplitude")
        axs[i].set_xlabel("Time (s)")
        # plot with a thin line
        axs[i].plot(
            np.arange( 0, 3000 ) / 100,
            bands[i].reshape( -1 ),
            linewidth=0.5,
            color="black",
            alpha=0.5,
        )
    
    plt.tight_layout()

    output_dir = f"output/model/{net}/{dataset}/plot/nrem2/{index}/"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(
        output_dir,
        f"bands.png"
    )    
    
    plt.savefig(output_path, dpi=300)

    plt.close(fig)
    
    
    
    
    
def plot_spectral_gradients(
    index : int,
    signal : np.ndarray,
    preds : np.ndarray,
    sg : np.ndarray,
    w : np.ndarray,    
    
):
        
    fig, axs = plt.subplots(3, 2, figsize=(12, 6), width_ratios=[1, 8] ) 
    fig.delaxes(axs[ 0, 0] )
    fig.delaxes(axs[ 2, 0] )

    pred = np.argmax( preds.reshape( 5 ))
    labels = [ "Wake", "NREM1", "NREM2", "NREM3", "REM" ]

        
    axs[0, 1].set_title( f"{labels[pred]} Signal - Confidence: {preds[pred]:.2f}")

    axs[0, 1].set_ylabel("Amplitude")
    # axs[0, 1].set_xlabel("Time (s)")
    # plot with a thin line
        
    axs[0, 1].plot(
        np.arange( 0, 3000 ) / 100,
        signal.reshape( -1 ),
        linewidth=0.5,
        color="black",
        alpha=0.5,
    )

    # remove the xticks
    axs[0, 1].set_xticks([])
    
    axs[0, 1].spines['top'].set_visible(False)
    axs[0, 1].spines['right'].set_visible(False)
    
    # --> in time we want to go from 3000 to 29 
    SGxx_ = np.zeros( ( 6, 30 ) )
    for j in range( 6 ):
        SGxx_[5 - j] = sg[j].reshape(30, 100).sum( axis = -1 )
        
    SGxx = np.zeros( (129, 30) )

    for i in range( 10 ):
        SGxx[ i, :] = SGxx_[0, :]
        SGxx[ i + 10, :] = SGxx_[1, :]    
        SGxx[ i + 20, :] = SGxx_[2, :]
        
        if i < 6:
            SGxx[ 30 + i, :] = SGxx_[3, :]
    
    for i in range( 36, 129 ):
        if i < 77:
            SGxx[ i, :] = SGxx_[4, :]
        else:
            SGxx[ i, :] = SGxx_[5, :]
                        
    
    axs[1, 1].set_title("Spectral Gradients")
    #axs[1, 1].set_ylabel("Frequency (Hz)")
    #axs[1, 1].set_xlabel("Time (s)")
    
        
    f = np.linspace(0, 50, 129)  # ad esempio fino a 50Hz
    t = np.arange( 0, 3000 ) / 100
    
    epoch_W = SGxx.sum( axis =  1 ).reshape( 129 )
    
    SGxx = SGxx.reshape( 129, 30, 1 ).repeat( 100, axis = -1 ).reshape( 129, 3000 )
    
    # Meshgrid per il colormesh
    #t, f = np.meshgrid(time_edges, freqs)

    #extent = [0, 30, 0, 50]
    axs[1, 1].pcolormesh(
        t,
        f,
        SGxx,
        shading="gouraud",
        cmap="viridis",
        #extent=extent,
    )
    
    # remove the xticks
    axs[1, 1].set_xticks([])
    
    axs[1, 1].spines['top'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)

    
    # now we plot the time extracted spectral gradients    
    TSG = sg.sum( axis = 0 ).reshape( 3000 )    
    axs[2, 1].set_title( f"TSG")
    axs[2, 1].set_ylabel("Importance")
    axs[2, 1].set_xlabel("Time (s)")
    # plot with a thin line
        
    axs[2, 1].plot(
        np.arange( 0, 3000 ) / 100,
        TSG.reshape( -1 ),
        linewidth=0.5,
        color="red",
        alpha=0.5,
    )
    axs[2, 1].spines['top'].set_visible(False)
    axs[2, 1].spines['right'].set_visible(False)

    # now we compute the epoch W 
    
    axs[1, 0].set_title( f"W")
    axs[1, 0].set_ylabel("Frequency (Hz)")
    axs[1, 0].set_xlabel("Importance")
    # plot with a thin line
        
    axs[1, 0].plot(
        epoch_W,
        f,
        linewidth=0.5,
        color="red",
        alpha=0.5,
    )
    
    axs[1, 0].spines['top'].set_visible(False)
    axs[1, 0].spines['right'].set_visible(False)

        
    plt.tight_layout()

    output_dir = f"output/model/{net}/{dataset}/plot/nrem2/{index}/"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(
        output_dir,
        f"spectral_gradients.png"
    )    
    
    plt.savefig(output_path, dpi=300)

    plt.close(fig)
    
def path_integral(
    fn: nn.Module,
    path: callable,
    jac_path: callable = None,
    n_points: int = 50,
):

    p = path(0)

    T = torch.linspace(0, 1, n_points, device=path(0).device, dtype=p.dtype)

    SG = 0

    curve = []

    for t in T:
        inputs = path(t).clone().detach().requires_grad_( True )
        outputs = fn( inputs ).squeeze()
        
        G = torch.autograd.grad(outputs, inputs, retain_graph=True)[0]
        curve = curve + [G]

    curve = torch.stack(curve, dim=0)
    curve = torch.trapezoid(curve, dx=1 / (n_points - 1), dim=0)
    
    J = jac_path(t)
    curve = torch.einsum("bn,bn->bn", curve, J)
        
    return curve

def fline_integral(
    fn: nn.Module,
    x_1: torch.Tensor,
    x_0: torch.Tensor,
    n_points: int = 50,
):
    def path(alpha: torch.Tensor):
        return x_0 - alpha * (x_0 - x_1)

    def jac_path(alpha: torch.Tensor):
        return -(x_0 - x_1)

    return path_integral(fn=fn, path=path, jac_path=jac_path, n_points=n_points)

def spectral_integral(
    fn : nn.Module,
    signal : torch.Tensor,
    n_points : int = 50,
    ):
    
    bands = isolate_bands( signal.detach().cpu().numpy() ) # shape (7, 1, 21, 3000)
    bands = torch.tensor( bands, dtype = signal.dtype, device = device )    
    
    SG, W, eps = [], [], []
    
    for i in range( 1, bands.shape[0] ):
        bands_importance = fline_integral(fn, bands[ i-1].reshape(1, -1), bands[ i ].reshape(1, -1), n_points ).reshape( signal.shape )
        SG.append( bands_importance )
        
        w = torch.abs(fn( bands[i-1]) - fn(bands[i]))
        W.append( w )
        
        w_ = torch.abs( torch.sum( bands_importance.reshape(-1) ) )
        
        eps.append( w - w_ )        
            
    SG = torch.stack( SG, dim = 0 ).reshape( 6, 21, 1, 3000 )
    W = torch.stack( W, dim = 0 ).reshape( -1 )
    
    W = torch.nn.functional.relu( W ) # 256, 21, 5
    W = torch.round( W, decimals = 2 )
    W = W.reshape(6) / W.reshape(6).sum()

    # W = torch.nn.functional.softmax( W, dim = -1 ) 
    
    # W_norm = torch.nn.functional.relu( fn( bands[0] ) - fn(bands[-1]) ) + 1e-8
    SG = torch.nn.functional.relu( SG ) # 256, 21, 5

    # W = W / W_norm
       
    eps = torch.stack( eps, dim = 0 ).reshape( -1 )    
    
    SG = torch.einsum("wst,w->wst", SG.reshape(6, 21, 3000), W.reshape(6))
    
    return SG.detach().cpu().numpy(), W.detach().cpu().numpy(), eps.detach().cpu().numpy()
    
def fast_W( fn: nn.Module, x: torch.Tensor ):
    # x_shape ( batch_size, 21, 1, 3000 )
    
    x = isolate_bands( x.detach().cpu().numpy() ) # shape (7, 1, 21, 3000)
    x = torch.tensor( x ).float()
    
    W = []
    for i in range( 1, x.shape[0] ):
        x_start = x[i-1].to(device)
        x_end = x[i].to(device)

        w = torch.abs(fn( x_start ) - fn(x_end)).detach().cpu()
        W.append( w )

        del x_start, x_end

    W = torch.stack( W, dim = 0 ).to(device) 

    x_start = x[0].to(device)
    x_end = x[-1].to(device)
    #W_norm = fn( x_start ) - fn(x_end) # 256, 21, 5

    del x_start, x_end
    
    W = torch.nn.functional.relu( W ) #
    W = torch.round( W, decimals = 2 )
    W = torch.permute( W, (1, 2, 3, 0 ) )

    return W.detach().cpu().numpy()
    
class ExWrapper(torch.nn.Module):
    def __init__(
        self,
        func: torch.nn.Module,
        index = None,
        target = None,
    ):
        super().__init__()
        self.model = func
        self.target = target
        self.index = index

    def forward(self, x):
        if len( x.shape ) > 1:
            batch_size = x.shape[0]
            batched = True
        else:
            batched = False
            batch_size = 1
            
        x = x.reshape(batch_size, 21, 1, 3000)

        x = self.model(x)

        if self.index is not None and self.target is not None:
            x = x[:, self.index, self.target]
        elif self.index is not None:
            x = x[:, self.index]
        elif self.target is not None:
            x = x[:, :, self.target]    
        else:
            pass

        if batched:
            x = x.reshape(batch_size, -1)
        else:
            x = x.reshape(-1)
            
        return x



if __name__ == "__main__":

    seed_everything(42, workers=True)
    set_float32_matmul_precision("medium")

    basedir = os.path.join( os.getcwd(), "output", "model", net, dataset )

    if not os.path.exists(basedir):
        os.makedirs(basedir)
        print(f"Creating directory {basedir}")

    ckpt_path = os.path.join( basedir, "checkpoint.ckpt")

    print( basedir )
    print( ckpt_path )

    model_kwargs = {
        "n_classes": 5,
        "sf" : 100,
        "in_channels": 1,
        "sequence_length" : 21,

        "loss" : "physioex.train.networks.utils.loss:CrossEntropyLoss",
        "loss_kwargs": {},
        
        "learning_rate" : .0001,
        "weight_decay" :  .000001,
    }

    model = TinySleepNet.load_from_checkpoint( ckpt_path, config = model_kwargs ).eval()

    model.nn.clf.rnn = model.nn.clf.rnn.train()
    
    model = torch.nn.Sequential(
        model,
        torch.nn.Softmax(dim=-1)
    )    
    
    with torch.no_grad():
        model = model.to(device)

        zero_pred = model( torch.zeros( 1, 21, 1, 3000).float().to(device) ).reshape( 21, -1 )
        zero_pred = torch.nn.functional.softmax( zero_pred, dim = -1 ).cpu().numpy().mean( axis = 0 )
        print( f"Model probabilities on zero: {zero_pred}")
        print( f"Model predictions on zero: {np.argmax(zero_pred)}")

        zero_pred = model( torch.randn( 1, 21, 1, 3000).float().to(device) ).reshape( 21, -1 )
        zero_pred = torch.nn.functional.softmax( zero_pred, dim = -1 ).cpu().numpy().mean( axis = 0 )
        print( f"Model probabilities on random: {zero_pred}")
        print( f"Model predictions on random: {np.argmax(zero_pred)}")
        
        model = model.to("cpu")


    datamodule = PhysioExDataModule(
            datasets = [dataset],
            batch_size = batch_size,
            data_folder = data_folder,
            num_workers = 7,
    )

    test_loader = datamodule.test_dataloader()

    # evaluate the model on the test set
    # evaluate_model( model, test_loader, basedir )
    
    # iterate over the test set 
    X_test, y_test, y_preds, w_test = [], [], [], []
    model = model.to(device)
    for i, (x, y) in enumerate( tqdm( test_loader ) ):    
        # x shape ( 256, 21, 1, 3000 ): batch size, sequence length, channels, time
        # y shape ( 256, 21 ): batch size, sequence length        
        
        y_hat = model( x.float().to(device) ).detach().cpu().numpy()

        X_test.append( x.cpu().numpy() )
        y_test.append( y.cpu().numpy() )
        y_preds.append( y_hat )
        w_test.append( fast_W( model, x ) )
                
    model = model.to("cpu")
        
    X_test = np.concatenate( X_test, axis = 0 ) # batch, 21, 1, 3000
    y_test = np.concatenate( y_test, axis = 0 ) # batch, 21
    y_preds = np.concatenate( y_preds, axis = 0 ) # batch, 21, 5       
    w_test = np.concatenate( w_test, axis = 0 ) # batch, 21, 5, 6
        
    nrem2 = X_test.reshape( -1, 3000 )
    indexes = np.arange( nrem2.shape[0] )
    
    nrem2_preds = np.argmax(y_preds.reshape(-1, 5), axis=-1)
    nrem2_labels = y_test.reshape(-1)
    nrem2_w = w_test.reshape( -1, 5, 6 )[:, 2]
    true_positive = np.where((nrem2_labels == 2) & (nrem2_preds == 2))[0]

    nrem2 = nrem2[ true_positive ]
    nrem2_labels = nrem2_labels[ true_positive ]
    nrem2_proba = y_preds.reshape(-1, 5)[:, 2][ true_positive ]
    nrem2_preds = nrem2_preds[ true_positive ]
    nrem2_w = nrem2_w[ true_positive ]
    indexes = indexes[ true_positive ]

    # we want to select the epochs for which the model is most confident -> proba > 0.85
    
    true_positive = np.where( nrem2_proba > 0.85 )[0]
    nrem2 = nrem2[ true_positive ]
    nrem2_labels = nrem2_labels[ true_positive ]
    nrem2_proba = nrem2_proba[ true_positive ]
    nrem2_preds = nrem2_preds[ true_positive ]
    nrem2_w = nrem2_w[ true_positive ]
    indexes = indexes[ true_positive ]

    print("Number of epochs with proba > 0.95: ", nrem2.shape[0])    

    # take the elements for with the sigma band ( 2 in 6 bands ) is the most important band i.e. nrem2_w[:, 2] > 0.8
    #sigma = np.where( nrem2_w[:, 2] > 0.8 )[0]
    
    sigma = np.argmax( nrem2_w, axis = -1 ) == 2
    if len(sigma) == 0:
        print("No epochs found with sigma band as most important band")
        exit()
    else:
        nrem2 = nrem2[sigma]
        nrem2_w = nrem2_w[sigma]
        nrem2_preds = nrem2_preds[sigma]
        true_positive = true_positive[sigma]  # Update true_positive to match filtered epochs
        indexes = indexes[sigma]  
        print("Number of epochs with sigma band as most important band: ", nrem2.shape[0])

    model = model.to(device)

    # store signals in a list to avoid plotting repetitions
    signals = []    
    for index in tqdm( indexes ):
            
        # index = indexes[0]
        sequence_indx = index  // 21
        epoch_indx = index % 21
        # print("Epoch index: ", epoch_indx, "Sequence index: ", sequence_indx)

        exmodel = ExWrapper(
            func = model,
            index = epoch_indx,
            target = 2,
        ).to( device )
        
        signal = torch.tensor(X_test[sequence_indx], dtype=torch.float32).reshape( 1, -1 ).to(device)
        # output = exmodel(signal).detach().cpu().item()
        # print the model output on the signal
        # print( "Model output: ",  output )
        
        SG, W, eps = spectral_integral( exmodel, signal, 100 )
        #print("SG shape: ", SG.shape, "W shape: ", W.shape, "eps shape: ", eps.shape)    
        eps = eps.reshape( -1 )
                
        preds = model( signal.reshape(1, 21, 1, 3000) ).reshape( 21, 5 )[epoch_indx].detach().cpu().numpy()
                
        # we want to compute the epoch in the sequence for which the sigma band is the most important band
        
        SG_indx = np.argmax( SG.reshape( 6, 21, 3000 )[2].sum( axis = -1).reshape( 21 ) ) 
        
        signal = signal.reshape( 21, 3000 )[SG_indx].detach().cpu().numpy()
        
        SG = SG.reshape( 6, 21, 3000 )[:, SG_indx]
        
        # now check that W_epoch has still sigma as the most important band
        W_epoch = SG.sum( axis = -1 ).reshape(6)
        
        if np.argmax( W_epoch ) != 2:
            continue
                
        W = W.reshape( -1 )
        
        # print( "Epsilon: ", eps.reshape(-1) )
        
        is_in_list = any(np.array_equal(signal, x) for x in signals)        
        if not is_in_list:
        
            plot_spectral_gradients(
                index = index,
                signal = signal,
                preds = preds,
                sg = SG,
                w = W,
            )
            
            plot_bands(
                index = index,
                signal = signal,
                preds = preds,
            )
        
            signals.append( signal )
            
        else:
            continue