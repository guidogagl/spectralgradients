import sys, os
sys.path.append("./")

from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from lightning.pytorch import seed_everything

import torch
from torch import nn

from loguru import logger

from torchmetrics import Accuracy

import pandas as pd
import os

from src.synt_data import SyntDataset, SETUPS
from src.train.model import TimeModule
from src.posthoc.sgradients import SpectralGradients

from torch.utils.data import SubsetRandomSampler, DataLoader

class Wrapper(torch.nn.Module):
    def __init__(self, nn, input_shape):
        super().__init__()
        self.nn = nn
        self.softmax = torch.nn.Softmax(dim=-1)
        self.input_shape = input_shape

    def forward(self, x):
        batched = True
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            batched = False

        # logger.info( x.shape )

        batch_size, n = x.shape
        y = self.nn(x.reshape(batch_size, *self.input_shape))

        y = self.softmax(y).reshape(batch_size, -1)

        if not batched:
            return y.reshape(-1)

        return y



torch.set_float32_matmul_precision('medium')

batch_size = 256
setups = [0, 1, 2]

n_points = 10
nperseg = 100
fs = 100



class Localization(nn.Module):
    def __init__(self, f: nn.Module = None, exp: nn.Module = None, name: str = "tfloc"):
        super(Localization, self).__init__()
        self.name = name

    @torch.no_grad()
    def forward(self, x, attr, mask):
        W, SG = attr
        attr = torch.einsum( "bcf,bcft->bcft", W, SG ) # attr shape : ( batch_size, classes, frequency_bins,  timestamps)
        
        # x shape : ( batch, timestamps )
        # mask shape : ( batch_size, timestamps, frequency_bins )

        batch_size, classes, timestamps, frequency_bins = attr.shape
        
        attr = torch.einsum( "bmft,bn->bmft", attr, torch.sign(x))
        attr = torch.nn.functional.relu(attr)
        
        Rtot = attr.reshape( batch_size, classes, -1 ).sum( -1 ) # b, m        
        Rtot = torch.where(Rtot > 0, Rtot, float("inf"))

        Rin = torch.einsum("bmft,bft->bm", attr, mask.float())
        mu = torch.einsum("bm,bm->bm", Rin, 1 / Rtot)

        Stot = torch.ones_like( mask ).reshape(batch_size, -1).sum( -1 ) # b
        Sin = mask.reshape(batch_size, -1).sum( -1 )

        return torch.einsum("bm,b->bm", mu, (Stot / Sin))


# let's define a stopband filter in scipy with 1 Hz Bandwidth and cutoff passed as argument
from scipy.signal import butter, filtfilt

def butter_bandstop(x, cutoff, fs = 100, order=5):
    # x is a torch tensor let's convert it to numpy
    device = x.device
    dtype = x.dtype
    x = x.reshape(-1).detach().cpu().numpy()
    cutoff = [ cutoff - 1, cutoff + 1 ]
    
    if cutoff[0] <= 0 :
        cutoff[0] = 0.1

    if cutoff[1] >= fs / 2:
        cutoff[1] = fs / 2 - 0.1
    
    
    if cutoff[0] < 0:
        print("cutoff[0] < 0, ", cutoff[0])
    if cutoff[1] > fs / 2:
        print("cutoff[1] > fs / 2, ", cutoff[1])
            
    b, a = butter(order, cutoff, fs = fs, btype='bandstop')
    y = filtfilt(b, a, x).copy()
    
    return torch.tensor( y, dtype = dtype, device=device)

def stopband_filter(x, w, index, fs):
    # x shape is ( batch_size, classes, timestamps )    
    # w shape is ( batch_size, classes, frequencies ) ordered from the most relevant to the less relevant
    
    # index is the int index of the frequency to remove
    
    x_filt = x.clone() 
    
    for i in range(x.shape[0]):
        for c in range(x.shape[1]):
            # get the frequency to remove
            freq = w[i, c, index].detach().cpu().numpy()
            # remove the frequency from the signal            
            x_filt[i, c] = butter_bandstop(x[i, c], freq, fs = fs)
            
    return x_filt
    
class Infidelity(nn.Module):
    def __init__(self, f: nn.Module = None, exp: nn.Module = None, name: str = "tfinf"):
        super(Infidelity, self).__init__()
        self.name = name
        self.f = f

    @torch.no_grad()
    def forward(self, x, attr, mask):
        W, _ = attr 

        # W shape is ( batch_size, classes, frequencies ) 
        # W is the relevance of the frequencies for each class
        # fs is 100 Hz with 1Hz resolution  frequencies are in the range [0, 50]
        # we want to get the array of the frequencies order by relevance
        batch_size, timestamps = x.shape
        classes = W.shape[1]
        imp_W = torch.argsort(W, dim=-1, descending=True) # b, m, f
        infidelity = [self.f(x)]  # b, m

        zeros = torch.zeros_like(x)
        x = [ x.clone() for _ in range( classes ) ] 
        
        x = torch.stack( x, dim = 1 ).to( W.device ) # b, m, timestamps

        for f in range(W.shape[-1]):
            # get the filtered signal
            x_filt = stopband_filter(x, imp_W, f, fs)
            
            # x_filt shape is ( batch_size, classes, timestamps )
            fx = self.f(x_filt.reshape( -1, timestamps) ).reshape( batch_size, classes, classes )
            
            # we want to get only the elemnt on the diagonal classes classes
            fx = torch.diagonal(fx, dim1=-2, dim2=-1)
            
            infidelity.append( fx )
            
            x = x_filt.clone()
        
        infidelity.append( self.f( zeros ) )
        infidelity = torch.stack(infidelity, dim=0)
        
        infidelity = torch.trapezoid( infidelity, dx = 1/infidelity.shape[0], dim = 0)
        return infidelity
                
class Complexity( nn.Module ):
    def __init__(self, f: nn.Module = None, exp: nn.Module = None, name: str = "tfcomp"):
        super(Complexity, self).__init__()
        self.name = name
    
    @torch.no_grad()
    def forward(self, x, attr, mask):
        
        W, SG = attr
        attr = torch.einsum( "bcf,bcft->bcft", W, SG )
        
        batch_size, classes, frequency_bins, timestamps = attr.shape
        C_i = torch.abs(attr).reshape( batch_size, classes, -1 ) + 1e-8
        C = C_i.sum(dim=-1)  # batch_size, m
        C_i = torch.einsum("bmn,bm->bmn", C_i, 1 / C)
        C = -torch.einsum("bmn,bmn->bm", C_i, torch.log(C_i))
        return C
        
def convert_to_tf( mask: torch.Tensor, targets : torch.Tensor, setup : int ):
    # convert a mask in time of shape batch_size, timestamps 
    # to a mask in frequency of shape batch_size, timestamps, frequency_bins

    # the relevant frequencies are specified by the setup and the target class
    
    batch_size, timestamps = mask.shape
    targets = targets.long()
    
    freq_res = fs / nperseg
    nyquist = fs / 2
    freqs = torch.arange(0, nyquist, freq_res).long()
    
    tf_mask = torch.zeros( batch_size, len(freqs), timestamps )   
    
    for i in range(batch_size):
        # get the relevant frequencies for the target class
        relevant_freqs = SETUPS[setup]["desc"][targets[i]][0]["freq"]       
        
        # 100 freq bins 1 freq resolution
        low = int( relevant_freqs - 5)
        high = int( relevant_freqs + 5)
        
        # get the index of the first and the last "1" the the time mask
        indexes = torch.where(mask[i] == 1)[0]
        low_time = indexes[0]
        high_time = indexes[-1]
        
        tf_mask[i, freqs[(freqs >= low) & (freqs <= high)], low_time:high_time + 1] = 1
        #tf_mask[i] = tf_mask[i].flip(0)
                   
    return tf_mask.to(mask.device)
    
class EvaluationModule(pl.LightningModule):
    def __init__(
        self,
        nn: torch.nn.Module,
        setup : int,
        input_shape: tuple,
        strategy: str = "highest",
        result_path : str = "output/model/"        
    ):
        super(EvaluationModule, self).__init__()

        self.result_path = f"{result_path}/setup{setup}/metrics/sg/{strategy}/"
        os.makedirs(self.result_path, exist_ok=True)
        
        self.nn = Wrapper( nn, input_shape )
        
        self.setup_indx = setup

        self.explainer = SpectralGradients(
            f = self.nn,
            fs = fs,
            Q = 5,
            nperseg = nperseg,
            strategy = strategy,
            n_points = n_points,
        )
        
        self.metrics = [
            Localization(), Infidelity( f = self.nn ), Complexity()
        ]
        
        self.acc = Accuracy(task="multiclass", num_classes=5)

    def forward(self, x, mask):

        W, SG = self.explainer(x)

        loc = self.metrics[0](x, (W, SG), mask)
        inf = self.metrics[1](x, (W, SG), mask)
        comp = self.metrics[2](x, (W, SG), mask)

        return ( loc, inf, comp)

    def validation_step(self, batch, batch_idx):
         # Logica di training
        inputs, mask, targets = batch
        mask = convert_to_tf(mask, targets, self.setup_indx)

        y_pred = self.nn(inputs)
        acc = self.acc(y_pred, targets)

        self.log(
            "val_acc",
            acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

        results = self.forward(inputs, mask)
        
        for result, name in zip( results, ["loc", "inf", "comp"] ):
            path = f"{self.result_path}/{name}/"
            os.makedirs( path, exist_ok = True)
                        
            result = torch.stack( [result[b, targets[b]] for b in range(len(result))], dim=0 )
            
            self.log(
                f"val_{name}",
                result.mean(),
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                sync_dist=True,
            )            
            
            if batch_idx == 0:
                torch.save( result.detach().cpu().reshape(-1), f"{path}/{name}.pt")
            else:
                try:
                    back_results = torch.load( f"{path}/{name}.pt" )
                except:
                    logger.error( f"File {path}/{name}.pt not found" )
                    exit()
                back_results = torch.cat( [back_results, result.detach().cpu().reshape(-1) ] )
                torch.save( back_results, f"{path}/{name}.pt")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, mask, targets = batch
        mask = convert_to_tf(mask, targets, self.setup_indx)

        y_pred = self.nn(inputs)
        acc = self.acc(y_pred, targets)

        self.log(
            "test_acc",
            acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

        results = self.forward(inputs, mask)
        
        for result, name in zip( results, ["loc", "inf", "comp"] ):
            path = f"{self.result_path}/"
            os.makedirs( path, exist_ok = True)
                        
            result = torch.stack( [result[b, targets[b]] for b in range(len(result))], dim=0 )
            
            self.log(
                f"test_{name}",
                result.mean(),
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                sync_dist=True,
            )            
            
            if batch_idx == 0:
                torch.save( result.detach().cpu().reshape(-1), f"{path}/{name}.pt")
            else:
                try:
                    back_results = torch.load( f"{path}/{name}.pt" )
                except:
                    logger.error( f"File {path}/{name}.pt not found" )
                    exit()
                back_results = torch.cat( [back_results, result.detach().cpu().reshape(-1) ] )
                torch.save( back_results, f"{path}/{name}.pt")
                
        

def evaluation_script(
    nn: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    setup: int,
    input_shape: tuple,
    strategy: str,
    ):

    seed_everything(42, workers=True)

    eval_module = EvaluationModule(
        nn = nn,
        setup = setup,
        input_shape = input_shape,
        strategy = strategy,
    )
    
    checkpoint_path = f"output/model/setup{setup}/metrics/sg/{strategy}/"

    trainer = pl.Trainer(
        devices="auto",
        strategy="auto",
        num_nodes=1,
        max_epochs=1,
        deterministic=True,
        inference_mode=False,
        logger=[
            TensorBoardLogger(save_dir=checkpoint_path),
            CSVLogger(save_dir=checkpoint_path),
        ],
    )

    results = trainer.test(eval_module, loader)
    
    results = pd.DataFrame( results )
    results.to_csv( checkpoint_path + "metrics.csv" )


if __name__ == "__main__":

    for setup in setups:

        data = SyntDataset(setup = setup, nperseg=nperseg, return_mask=True)

        loader = DataLoader(
            data,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(list(range(len(data) - data.n_samples))),
            num_workers=os.cpu_count(),
        )

        nn = TimeModule.load_from_checkpoint(
            f"output/model/setup{setup}/checkpoint.ckpt",
            input_shape=data[0][0].shape,
            fs=100,
            n_classes=data.n_class,
        ).nn.eval()


        for strategy in ["highest", "lowest"]:
            evaluation_script(
                nn=nn,
                loader=loader,
                setup=setup,
                input_shape=data[0][0].shape,
                strategy=strategy,
            )