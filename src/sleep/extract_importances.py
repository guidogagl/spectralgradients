import os 
import matplotlib.pyplot as plt
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

from src.sleep.utils import evaluate_model, get_importances, data_folder
from src.sleep.train import TinyTransformerNet
from src.posthoc.sgradients import SpectralGradients

class Wrapper( torch.nn.Module ):
    def __init__( self, model ):
        super(Wrapper, self).__init__()
        
        self.model = torch.nn.Sequential(
            model,
            torch.nn.Softmax(dim=-1)
        )    
    def forward( self, x ):
        if debiased:
            return self.model( x )[..., :-1]
        return self.model( x )

if __name__ == "__main__":
    seed_everything(42, workers=True)
    set_float32_matmul_precision("medium")

    batch_size = 32
    debiased = False
    dataset = "sleepedf"
    net = "tinytransformer"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    basedir = os.path.join( os.getcwd(), "output", "model", net, dataset )

    if debiased:
        basedir = os.path.join( basedir, "debiased")
    else:
        basedir = os.path.join( basedir, "standard")

    if not os.path.exists(basedir):
        os.makedirs(basedir)
        print(f"Creating directory {basedir}")

    ckpt_path = os.path.join( basedir, "checkpoint.ckpt")
    #ckpt_path = f"{basedir}checkpoint.ckpt"

    print( basedir )
    print( ckpt_path )
    
    model_kwargs = {
    "n_classes": 5,
    "sf" : 100,
    "in_channels": 1,
    "sequence_length" : 21,

    "loss" : "physioex.train.networks.utils.loss:CrossEntropyLoss",
    "loss_kwargs": {},
    "debias" : debiased,
    
    "learning_rate" : .0001,
    "weight_decay" :  .000001,
    }

    model = TinyTransformerNet.load_from_checkpoint( ckpt_path, config = model_kwargs ).eval()
    # add a softmax layer to the model

    # evaluate the model on the background data

    with torch.no_grad():
        steps = 20
        y_pred = []
        model = model.to(device)
        
        zero_pred = model( torch.zeros( 1, 21, 1, 3000).float().to(device) ).reshape( 21, -1 )
        zero_pred = torch.nn.functional.softmax( zero_pred, dim = -1 ).cpu().numpy().mean( axis = 0 )
        print( f"Model probabilities on zero: {zero_pred}")
        print( f"Model predictions on zero: {np.argmax(zero_pred)}")
        
        model = model.to("cpu")


    model = Wrapper( model = model)

    datamodule = PhysioExDataModule(
            datasets = [dataset],
            batch_size = batch_size,
            data_folder = data_folder,
            num_workers = os.cpu_count(),
    )

    test_loader = datamodule.test_dataloader()

    evaluate_model( model, test_loader, basedir )

    # get the importances "x", the stages "y" and the predictions "y_hat"
    x_train, y_train, y_hat_train, x_val, y_val, y_hat_val, x_test, y_test, y_hat_test = get_importances( model, datamodule, basedir)