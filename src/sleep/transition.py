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
from src.sleep.sleep_spindles import ExWrapper, fast_W, spectral_integral
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

    output_dir = f"output/model/{net}/{dataset}/plot/transition/{index}/"
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
    sg2 : np.ndarray,
    w2 : np.ndarray,    
    sg3 : np.ndarray,
    w3 : np.ndarray, 
):
        
    fig, axs = plt.subplots(3, 2, figsize=(12, 6), width_ratios=[1, 8] ) 
    fig.delaxes(axs[ 0, 0] )
        
    axs[0, 1].set_title( f"Confidence NREM2: {preds[2]:.2f} - NREM3: {preds[3]:.2f}" )

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
        SGxx_[5 - j] = sg3[j].reshape(30, 100).sum( axis = -1 )
        
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
                        
    
    axs[1, 1].set_title("Spectral Gradients NREM3")
        
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
    
    
    SGxx_ = np.zeros( ( 6, 30 ) )
    for j in range( 6 ):
        SGxx_[5 - j] = sg2[j].reshape(30, 100).sum( axis = -1 )
        
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
                        
    
    axs[2, 1].set_title("Spectral Gradients NREM2")
        
    f = np.linspace(0, 50, 129)  # ad esempio fino a 50Hz
    t = np.arange( 0, 3000 ) / 100
    
    epoch_W = SGxx.sum( axis =  1 ).reshape( 129 )
    
    SGxx = SGxx.reshape( 129, 30, 1 ).repeat( 100, axis = -1 ).reshape( 129, 3000 )
    
    axs[2, 1].pcolormesh(
        t,
        f,
        SGxx,
        shading="gouraud",
        cmap="viridis",
        #extent=extent,
    )
    
    axs[2, 1].set_xticks([])
    
    axs[2, 1].spines['top'].set_visible(False)
    axs[2, 1].spines['right'].set_visible(False)
    
    axs[2, 0].set_title( f"W")
    axs[2, 0].set_ylabel("Frequency (Hz)")
    axs[2, 0].set_xlabel("Importance")
        
    axs[2, 0].plot(
        epoch_W,
        f,
        linewidth=0.5,
        color="red",
        alpha=0.5,
    )
    
    axs[2, 0].spines['top'].set_visible(False)
    axs[2, 0].spines['right'].set_visible(False)    
        
    plt.tight_layout()

    output_dir = f"output/model/{net}/{dataset}/plot/transition/{index}/"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(
        output_dir,
        f"spectral_gradients.png"
    )    
    
    plt.savefig(output_path, dpi=300)

    plt.close(fig)
    


if __name__ == "__main__":

    seed_everything(42, workers=True)
    set_float32_matmul_precision("medium")

    basedir = os.path.join( os.getcwd(), "output", "model", net, dataset )

    if not os.path.exists(basedir):
        os.makedirs(basedir)
        print(f"Creating directory {basedir}")

    ckpt_path = os.path.join( basedir, "checkpoint.ckpt")

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
        
    nrem3 = X_test.reshape( -1, 3000 )
    indexes = np.arange( nrem3.shape[0] )
    
    nrem3_preds = np.argmax(y_preds.reshape(-1, 5), axis=-1)
    nrem3_labels = y_test.reshape(-1)
    nrem3_w = w_test.reshape( -1, 5, 6 )[:, 3]
    true_positive = np.where((nrem3_labels == 3) & (nrem3_preds == 3))[0]

    nrem3 = nrem3[ true_positive ]
    nrem3_labels = nrem3_labels[ true_positive ]
    nrem3_proba = y_preds.reshape(-1, 5)[:, 3][ true_positive ]
    nrem2_proba = y_preds.reshape(-1, 5)[:, 2][ true_positive ]
    nrem3_preds = nrem3_preds[ true_positive ]
    nrem3_w = nrem3_w[ true_positive ]
    indexes = indexes[ true_positive ]

    # we want to identify the epochs in a transition between nrem2 and nrem3
    # so the probability of nrem2 should be > 0.4 while the predition is nrem3
        
    true_positive = np.where( nrem2_proba > 0.4 )[0]
    nrem3 = nrem3[ true_positive ]
    nrem3_labels = nrem3_labels[ true_positive ]
    nrem3_proba = nrem3_proba[ true_positive ]
    nrem3_preds = nrem3_preds[ true_positive ]
    nrem3_w = nrem3_w[ true_positive ]
    indexes = indexes[ true_positive ]

    print("Number of epochs with nrem2 proba > 0.4: ", nrem3.shape[0])    

    
    model = model.to(device)

    # store signals in a list to avoid plotting repetitions
    signals = []    
    for index in tqdm( indexes ):
            
        # index = indexes[0]
        sequence_indx = index  // 21
        epoch_indx = index % 21
        
        signal = torch.tensor(X_test[sequence_indx], dtype=torch.float32).reshape( 1, -1 ).to(device)

        is_in_list = any(np.array_equal(signal, x) for x in signals)        
        if is_in_list:
            continue

        # we want now to plot the signal looking at nrem2 and nrem3    
        exmodel = ExWrapper(
            func = model,
            index = epoch_indx,
            target = 3,
        ).to( device )
        
        SG3, W3, _ = spectral_integral( exmodel, signal, 100 )
        
        exmodel = ExWrapper(
            func = model,
            index = epoch_indx,
            target = 2,
        ).to( device )
       
        SG2, W2, _ = spectral_integral( exmodel, signal, 100 )
                
        preds = model( signal.reshape(1, 21, 1, 3000) ).reshape( 21, 5 )[epoch_indx].detach().cpu().numpy()
                        
        SG2 = SG2.reshape( 6, 21, 3000 )[:, epoch_indx]
        SG3 = SG3.reshape( 6, 21, 3000 )[:, epoch_indx]
        
        # now check that W_epoch has still sigma as the most important band
        W2e = SG2.sum( axis = -1 ).reshape(6)
        W3e = SG3.sum( axis = -1 ).reshape(6) 
        
        signal = signal.reshape( 21, 3000 )[epoch_indx].detach().cpu().numpy()
        
        plot_spectral_gradients(
            index = index,
            signal = signal,
            preds = preds,
            sg2 = SG2,
            w2 = W2,
            sg3 = SG3,
            w3 = W3,
        )
        
        plot_bands(
            index = index,
            signal = signal,
            preds = preds,
        )
    
        signals.append( signal )
            