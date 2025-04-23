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
        
        self.pe = PositionalEncoding( 2048 * config["in_channels"] )

        t_layer = nn.TransformerEncoderLayer(
            d_model = 2048 * config["in_channels"],
            nhead=8*16,
            dim_feedforward=1024,
            dropout=0.5,
            batch_first=True,
        )
        
        self.encoder = nn.TransformerEncoder(
            t_layer, 
            num_layers=4
        )
        
        self.clf = nn.Linear(
            2048 * config["in_channels"], config["n_classes"]
        )

    def forward(self, x):
        x = self.pe(x)
        x = self.encoder(x)
        x = self.clf(x)
        return x

    def encode(self, x):
        x = self.pe(x)
        x = self.encoder(x)
        return x


def agument_data( batch ):
    inputs, targets = batch
    
    batch_size, seqlen, inchan, timestamps = inputs.shape
    
    # reshape the data to ignore the sequences
    
    inputs = inputs.reshape( batch_size * seqlen, inchan, timestamps )
    targets = targets.reshape( batch_size * seqlen )
    
    # select randomly 1/6 of the data to be background data
    indx = torch.randperm( inputs.shape[0] )[:inputs.shape[0] // 6]
    
    # randn -> random normal distribution [-3, 3]
    inputs[ indx ] = torch.randn( inputs[indx].shape ).to(inputs.device) * 0.1
    targets[ indx ] = 5 # background class    
    
    # reshape the data back to the original shape
    inputs = inputs.reshape( batch_size, seqlen, inchan, timestamps )
    targets = targets.reshape( batch_size, seqlen )
    
    return inputs, targets

class TinyTransformerNet( TinySleepNet ):
    def __init__(self, module_config):
        # add one class for the background data
        # 5 stages + 1 background
        module_config["n_classes"] = 6 
        
        super(TinyTransformerNet, self).__init__( module_config=module_config )
        self.nn.clf = TransformerClassifier( config=module_config )

    # redefine the train and validation step to add the background data 
    
    def training_step(self, batch, batch_idx):        
        return super().training_step( agument_data(batch), batch_idx)

    # def validation_step(self, batch, batch_idx):
    #    return super().validation_step(agument_data(batch), batch_idx)

if __name__ == "__main__":
    
    dataset = "sleepedf"
    batch_size = 32
     
    datamodule = PhysioExDataModule(
            datasets = [dataset],
            batch_size = batch_size,
            data_folder = os.environ["PHYSIOEXDATA"],
            num_workers = os.cpu_count(),
    )

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



    train_kwargs = {
        "datasets": datamodule,
        "model": TinyTransformerNet( model_kwargs ),
        "num_validations": 10,
        "checkpoint_path": f"output/model/tinytransformer/{dataset}/",
        "max_epochs": 20,
        "resume": True,
    }

    best_checkpoint = train(**train_kwargs)
    best_checkpoint = os.path.join(train_kwargs["checkpoint_path"], best_checkpoint)

    model = TinyTransformerNet.load_from_checkpoint( best_checkpoint, model_config = model_kwargs ).eval()

    test(
        datasets=datamodule,
        model=model,
        batch_size=32,
        results_path="output/model/tinytransformer/",
        aggregate_datasets= False,
    )
