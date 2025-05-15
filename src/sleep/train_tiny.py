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
    
    loss = loss + ( 0.01 * zero_pred )

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
    debias = True
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
        
        "learning_rate" : .0001,
        "weight_decay" :  .000001,
    }    
    
    train_kwargs = {
        "datasets": datamodule,
        "num_validations": 10,
        "max_epochs": 100,
    }

    if not debias:
        train_kwargs["checkpoint_path"] = f"output/model/tinysleepnet/{dataset}/standard/"
        train_kwargs["model"] = TinySleepNet( module_config = model_kwargs )
    else:
        train_kwargs["checkpoint_path"] = f"output/model/tinysleepnet/{dataset}/debiased/"        
        checkpoint = os.path.join(train_kwargs["checkpoint_path"], "checkpoint.ckpt")
        
        model_kwargs["learning_rate"] = .0000001
        model_kwargs["weight_decay"] =  .00000001
        
        # now we need to redefine the "compute_loss" function of the class TinySleepNet
        TinySleepNet.compute_loss = compute_loss
        train_kwargs["model"] = TinySleepNet.load_from_checkpoint( checkpoint, module_config = model_kwargs )
 
    # create the directory if it does not exist
    os.makedirs(train_kwargs["checkpoint_path"], exist_ok=True)
    
    best_checkpoint = train(**train_kwargs)
    best_checkpoint = os.path.join(train_kwargs["checkpoint_path"], best_checkpoint)

    model = TinySleepNet.load_from_checkpoint( best_checkpoint, module_config = model_kwargs ).eval()

    test(
        datasets=datamodule,
        model=model,
        batch_size=batch_size,
        results_path=train_kwargs["checkpoint_path"],
        aggregate_datasets= False,
    )
