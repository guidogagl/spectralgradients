import os
import uuid

import torch

from physioex.data import PhysioExDataModule, PhysioExDataset

from physioex.train.utils.test import test
from physioex.train.utils.train import train

from physioex.train.networks.tinysleepnet import TinySleepNet


class PriorSleepNet( TinySleepNet ):
    def __init__(self, module_config):
        super(PriorSleepNet, self).__init__( module_config=module_config )
        self.softmax = torch.nn.Softmax(dim=-1)

    def compute_priors( self, batch, outputs, log = "val" ):
        inputs, targets = batch
        batch_size, seqlen, inchan, nsamples = inputs.shape

        outputs = self.softmax(outputs) 
        
        _, gamma = self.encode( inputs[:, :, 1].reshape(batch_size, seqlen, 1, nsamples) )
        temp = self.softmax(gamma)
        gamma = outputs - temp
        outputs = temp

        _, beta = self.encode( inputs[:, :, 2].reshape(batch_size, seqlen, 1, nsamples) )
        temp = self.softmax(beta)
        beta = outputs - temp
        outputs = temp

        _, sigma = self.encode( inputs[:, :, 3].reshape(batch_size, seqlen, 1, nsamples) )
        temp = self.softmax(sigma)
        sigma = outputs - temp
        outputs = temp

        _, alpha = self.encode( inputs[:, :, 4].reshape(batch_size, seqlen, 1, nsamples) )
        temp = self.softmax(alpha)
        alpha = outputs - temp
        outputs = temp

        _, theta = self.encode( inputs[:, :, 5].reshape(batch_size, seqlen, 1, nsamples) )
        temp = self.softmax(theta)
        theta = outputs - temp
        outputs = temp

        _, delta = self.encode( torch.zeros_like(inputs[:, :, 1]).reshape(batch_size, seqlen, 1, nsamples) )
        temp = self.softmax(delta)
        delta = outputs - temp
        outputs = temp
        
        priors = torch.zeros_like( outputs )

        # wake priors
        priors[ ..., 0] = gamma[..., 0] + sigma[..., 0] + theta[..., 0] + delta[..., 0]

        # NREM 1 priors ( theta - alpha relevant )
        priors[ ..., 1] = gamma[..., 1] + sigma[..., 1] + beta[..., 1] + delta[..., 1]

        #NREM 2 priors ( theta - sigma relevant)
        priors[ ..., 2] = gamma[..., 2] + beta[..., 2] + alpha[..., 2] + delta[..., 2]

        #NREM 3 priors ( delta relevant )
        priors[ ..., 3] = gamma[...,3] + sigma[..., 3] + alpha[..., 3] + sigma[..., 3]

        # REM priors ( as wake )
        priors[ ..., 4] = gamma[..., 4] + sigma[..., 4] + theta[..., 4] + delta[..., 4]

        priors = torch.sum( torch.abs(priors), dim = -1).reshape( -1 ).mean()

        self.log( f"{log}_priors", priors, prog_bar=True, sync_dist=True )

        return priors

    def training_step(self, batch, batch_idx):
        if "val_loss" not in self.trainer.logged_metrics:
            self.log("val_loss", float("inf"))

        # Logica di training
        inputs, targets = batch
        batch_size, seqlen, inchans, timestamps = inputs.shape
        embeddings, outputs = self.encode(inputs[:, :, 0 ].reshape( batch_size, seqlen, 1, timestamps) )

        priors = self.compute_priors( batch, outputs, "train")

        return self.compute_loss(embeddings, outputs, targets) + priors

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        batch_size, seqlen, inchans, timestamps = inputs.shape
        embeddings, outputs = self.encode(inputs[:, :, 0 ].reshape( batch_size, seqlen, 1, timestamps) )

        priors = self.compute_priors( batch, outputs, "val")

        return self.compute_loss(embeddings, outputs, targets, "val") + priors

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch
        # input shape batch, seqlen, nchan, timestamps

        batch_size, seqlen, inchans, timestamps = inputs.shape
        embeddings, outputs = self.encode(inputs[:, :, 0 ].reshape( batch_size, seqlen, 1, timestamps) )

        return self.compute_loss(embeddings, outputs, targets, "test", log_metrics=True)

if __name__ == "__main__":
    
    dataset = PhysioExDataset(
        datasets = ["sleepedf"],
        data_folder = "output/data/",
        preprocessing = "isolate_bands",
        selected_channels = [ "EEG", "GAMMA", "BETA", "SIGMA", "ALPHA", "THETA", "DELTA" ],
        sequence_length = 21,
        indexed_channels = [ "EEG", "GAMMA", "BETA", "SIGMA", "ALPHA", "THETA", "DELTA" ],
    )

    datamodule = PhysioExDataModule(
        datasets = dataset,
        batch_size = 32,
        folds = -1
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
        "model": PriorSleepNet( model_kwargs ),
        "num_validations": 10,
        "checkpoint_path": "models/prior/",
        "max_epochs": 20,
        "resume": True,
        "num_validations" : 5,
    }

    best_checkpoint = train(**train_kwargs)
    best_checkpoint = os.path.join(train_kwargs["checkpoint_path"], best_checkpoint)

    model = PriorSleepNet.load_from_checkpoint( best_checkpoint, model_config = model_kwargs ).eval()

    test(
        datasets=datamodule,
        model=model,
        batch_size=32,
        results_path="models/prior/",
        aggregate_datasets= False,
    )
