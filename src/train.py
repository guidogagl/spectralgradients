import pytorch_lightning as pl 


# add . to the path

from src.model import TimeModule
from src.synt_data import SyntDataset, CLASS_DESC as PSD_FREQS

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler



import pandas as pd

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit 
from torch.utils.data import DataLoader, SubsetRandomSampler

import torch
from loguru import logger   

model_type = "conv"
# model_type = "conv"

def train():
    
    # set the seeds
    pl.seed_everything(42, workers = True)
    # set the precision of the cuda tensor cores
    torch.set_float32_matmul_precision('medium')

 
    logger.info("Starting the training script")
    
    logger.info("Building the dataset")
    dataset = SyntDataset( PSD_FREQS)
    
    # build the dataloader with .70 .15 .15 split

    logger.info("Building the dataloaders")
    # split the indexes in train, valid and test in a stratified way
# split the indexes in train, valid and test in a stratified way
    n_samples = len(dataset)
    n_train = int(n_samples * 0.7)
    n_valid = int(n_samples * 0.15)
    n_test = n_samples - n_train - n_valid

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_valid + n_test, random_state=42)
    train_indexes, temp_indexes = next(sss.split(np.zeros(n_samples), dataset.labels.numpy()))

    sss_valid_test = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=42)
    valid_indexes, test_indexes = next(sss_valid_test.split(np.zeros(len(temp_indexes)), dataset.labels.numpy()[temp_indexes]))

    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=32, sampler=SubsetRandomSampler(train_indexes), num_workers=8)
    valid_loader = DataLoader(dataset, batch_size=32, sampler=SubsetRandomSampler(valid_indexes), num_workers=8)
    test_loader = DataLoader(dataset, batch_size=32, sampler=SubsetRandomSampler(test_indexes), num_workers=8)

     
    logger.info("Building the model")
    # create the model
    model = TimeModule( model = model_type)
    
    # create the checkpoint callback
    
    callbacks = [ pl.callbacks.ModelCheckpoint(
        monitor = "val_acc",
        mode = "max",
        save_top_k = 1,
        filename="{epoch}-{step}-{val_acc:.2f}",
        save_weights_only=False,
        dirpath = "output/model/checkpoint",
    )]
    
    # create the csv logger 
    
    csvlogger = pl.loggers.CSVLogger("output/model/logs", name = "synt_model")
    
    # create the trainer
    trainer = pl.Trainer(
        max_epochs = 20,
        deterministic = True,
        callbacks = callbacks,
        logger = csvlogger,
        val_check_interval = 300,
    )

    logger.info("Starting the training")    
    
    # train the model
    trainer.fit(model, train_loader, valid_loader)
    
    # load the best model
    best_model = TimeModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model = model_type)
    
    # test the model
    
    test_results = trainer.test(best_model, test_loader)[0]
    test_results["split"] = "test"
    test_results = pd.DataFrame([test_results])
    
    valid_results = trainer.test(best_model, valid_loader)[0]
    valid_results["split"] = "valid"
    valid_results = pd.DataFrame([valid_results])
    
    train_results = trainer.test(best_model, train_loader)[0]
    train_results["split"] = "train"
    train_results = pd.DataFrame([train_results])
    
    # save the results in a csv file

    results = pd.concat([test_results, valid_results, train_results])
    
    results.to_csv( "synt_model/results.csv", index = False)
    
    return results, best_model
    

if __name__ == "__main__":
    
    train()