import os
import uuid

import torch

from physioex.data import PhysioExDataModule, PhysioExDataset

from physioex.train.utils.test import test
from physioex.train.utils.train import train

from physioex.train.networks.tinysleepnet import TinySleepNet
from physioex.train.networks.sleeptransformer import PositionalEncoding

from torch import nn
import os


module_config = {}
class TransformerClassifier(nn.Module):
    def __init__(self, config=module_config):
        super(TransformerClassifier, self).__init__()
        self.config = config
        
        self.reduce = nn.Linear( 2048, 128 )
        
        self.pe = PositionalEncoding( 128 )

        t_layer = nn.TransformerEncoderLayer(
            d_model = 128,
            nhead = 8,
            dim_feedforward=128,
            dropout=0.5,
            batch_first=True,
        )
        
        self.encoder = nn.TransformerEncoder(
            t_layer, 
            num_layers=4
        )
        
        self.clf = nn.Linear(
            128, config["n_classes"]
        )

    def forward(self, x):
        x = self.reduce(x)
        x = nn.functional.relu(x)
        
        x = self.pe(x)
        x = self.encoder(x)
        x = self.clf(x)
        return x

    def encode(self, x):
        x = self.reduce(x)
        x = nn.functional.relu(x)
        
        x = self.pe(x)
        x = self.encoder(x)
        return x

class TinyTransformerNet( TinySleepNet ):
    def __init__(self, module_config):
        # add one class for the background data
        # 5 stages + 1 background
            
        super(TinyTransformerNet, self).__init__( module_config=module_config )        

        self.nn.clf = TransformerClassifier( config=module_config )

        self.debias = module_config["debias"]

    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):

        batch_size, seq_len, n_class = outputs.size()

        embeddings = embeddings.reshape(batch_size * seq_len, -1)
        outputs = outputs.reshape(-1, n_class)
        targets = targets.reshape(-1)
        
        loss = self.loss(embeddings, outputs, targets)

        self.log(f"{log}_loss", loss, prog_bar=True, sync_dist=True)
        
        # now we compute the loss term in 0
        zero_pred = self( torch.zeros( 1, 21, 1, 3000).float().to( embeddings.device )).view(-1, n_class)
        zero_pred = torch.nn.functional.softmax( zero_pred, dim = -1 )[:, -1]
        self.log(f"{log}_zero_conf", zero_pred.mean(), prog_bar=True, sync_dist=True)

        # we want the output to converge to 0 hence -log(1 - x)
        zero_pred = - torch.log( zero_pred ).view(-1).mean()
        
        # loss = loss + zero_pred

        # now we want that for any other class in the problem the confidence of the background class is 0
        #zero_pred = torch.nn.functional.softmax( outputs, dim =-1 )[:, -1]
        #zero_pred = - torch.log( 1 - zero_pred ).view(-1).mean()
        
        #loss = loss + zero_pred
                
        self.log(f"{log}_acc", self.wacc(outputs, targets), prog_bar=True, sync_dist=True)
        
        if log_metrics and self.n_classes > 1:
            self.log(f"{log}_f1", self.wf1(outputs, targets), sync_dist=True)
            self.log(f"{log}_ck", self.ck(outputs, targets), sync_dist=True)
            self.log(f"{log}_pr", self.pr(outputs, targets), sync_dist=True)
            self.log(f"{log}_rc", self.rc(outputs, targets), sync_dist=True)
            self.log(f"{log}_macc", self.macc(outputs, targets), sync_dist=True)
            self.log(f"{log}_mf1", self.mf1(outputs, targets), sync_dist=True)
    
        return loss

    
if __name__ == "__main__":
    
    dataset = "sleepedf"
    batch_size = 128
    debias = False
    data_folder = "/scratch/leuven/365/vsc36564/"
    
    datamodule = PhysioExDataModule(
            datasets = [dataset],
            batch_size = batch_size,
            data_folder = data_folder,
            num_workers = os.cpu_count(),
    )

    model_kwargs = {
        "n_classes": 6,
        "sf" : 100,
        "in_channels": 1,
        "sequence_length" : 21,
    
        "loss" : "physioex.train.networks.utils.loss:CrossEntropyLoss",
        "loss_kwargs": {},
        "debias" : debias,

        "learning_rate" : .0001,
        "weight_decay" :  .000001,
    }
    
    
    
    train_kwargs = {
        "datasets": datamodule,
        "num_validations": 10,
        "max_epochs": 100,
    }

    if debias:
        train_kwargs["checkpoint_path"] = f"output/model/tinytransformer/{dataset}/debiased/"
        
    else:
        train_kwargs["checkpoint_path"] = f"output/model/tinytransformer/{dataset}/standard/"
    
    # create the directory if it does not exist
    os.makedirs(train_kwargs["checkpoint_path"], exist_ok=True)
    
    # check if there is already a checkpoint
    # list all files in the directory
    ckpt_files = [ ckpt for ckpt in os.listdir( train_kwargs["checkpoint_path"] ) if ckpt.endswith(".ckpt") ]
    if len( ckpt_files ) > 0:
        # load the model
        train_kwargs["model"] = TinyTransformerNet.load_from_checkpoint( os.path.join( train_kwargs["checkpoint_path"], ckpt_files[0] ), model_config = model_kwargs )
        print( f"Loading checkpoint {ckpt_files[0]} from {train_kwargs['checkpoint_path']}")
    else:
        if not debias:
            train_kwargs["model"] = TinyTransformerNet( model_kwargs )
        else:
            train_kwargs["model"] = TinyTransformerNet.load_from_checkpoint( f"output/model/tinytransformer/{dataset}/standard/checkpoint.ckpt", model_config = model_kwargs )
    
    best_checkpoint = train(**train_kwargs)
    best_checkpoint = os.path.join(train_kwargs["checkpoint_path"], best_checkpoint)

    model = TinyTransformerNet.load_from_checkpoint( best_checkpoint, model_config = model_kwargs ).eval()

    test(
        datasets=datamodule,
        model=model,
        batch_size=batch_size,
        results_path=train_kwargs["checkpoint_path"],
        aggregate_datasets= False,
    )
