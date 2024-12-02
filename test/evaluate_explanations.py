import numpy as np
import pandas as pd
from collections import OrderedDict

import torch
import torch.nn as nn

import sys, os
from tqdm.autonotebook import tqdm

from loguru import logger

import pytorch_lightning as pl

pl.seed_everything(1000)

sys.path.append(os.getcwd())



##### DATA LOADING #######
from src.synt_data import SyntDataset

logger.info("Loading dataset")


data = SyntDataset( return_mask=True)

from torch.utils.data import SubsetRandomSampler, DataLoader

batch_size = 256

loader = DataLoader( data, batch_size = batch_size, sampler = SubsetRandomSampler( list(range(len(data) - data.n_samples ) ) ))

######  MODEL LOADING ######

from src.train.model import TimeModule

logger.info("Loading model from checkpoint")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fn = TimeModule.load_from_checkpoint("output/model/checkpoint/epoch=19-step=21686-val_acc=0.93.ckpt", n_classes = data.n_class).eval()
fn = nn.Sequential(OrderedDict([("fn", fn), ("softmax", nn.Softmax(dim=-1))])).to(device)


# Explanations methods

from src.explainer_wrapper import explainer_wrapper, localization, lle, irof

logger.info("Model and data loaded, preparing explanation experiment...")

baseline_size = 4*batch_size
baseline = torch.randperm( len(data))[:baseline_size].long()

baseline, _, _ = data[ baseline ]

methods = [
    {
        "method": "WSpectralGradients",
    },
    {
        "method": "SpectralGradients",
    },
    {
        "method": "Saliency",
    },
    {
        "method": "InputXGrad",
    },
    {
        "method": "ExpectedGradients",
        "baseline": baseline,
    },
    {
        "method": "IntegratedGradients",
    },

]

for method in methods:

    method["model"] = fn

    method["loc"] = []
    method["irof"] = []
    method["lle"] = []


for batch in tqdm(loader):

    x_batch, m_batch, y_batch = batch

    for method in methods:

        method["model"] = fn

        method["inputs"] = x_batch
        method["targets"] = y_batch

        explainer, a_batch = explainer_wrapper(**method)

        if method["method"] == "SpectralGradients":
            explainer_ = explainer

            def explainer(x):
                explainer_.to(device)
                explanations = explainer_(x)

                explanations = explanations.sum(dim=2)
                explainer_.to("cpu")
                return explanations

        elif method["method"] == "WSpectralGradients":
            explainer_ = explainer

            def explainer(x):
                explainer_.to(device)
                explanations = explainer_(x)

                weights = torch.abs(explanations.sum(dim=-1))  # frequency weights
                explanations = explanations * weights.unsqueeze(-1)
                explanations = explanations.sum(dim=2)
                explainer_.to("cpu")
                return explanations

        else:
            pass

        logger.info(f"Evaluating {method["method"]} with LLE")
        method["lle"] += [
            lle(
                explainer=explainer,
                model=fn,
                x=x_batch,
                y=y_batch,
                attr=a_batch,
                mask=m_batch,
            ).item()
        ]

        logger.info(f"Evaluating {method["method"]} with IROF")
        method["irof"] += [
            irof(
                explainer=explainer,
                model=fn,
                x=x_batch,
                y=y_batch,
                attr=a_batch,
                mask=m_batch,
            ).item()
        ]

        logger.info(f"Evaluating {method["method"]} with Localization")
        method["loc"] += [
            localization(
                explainer=explainer,
                model=fn,
                x=x_batch,
                y=y_batch,
                attr=a_batch,
                mask=m_batch,
            ).item()
        ]

        np.savez(
            f"output/metrics/{method["method"]}.npz",
            irof=np.array(method["irof"]),
            lle=np.array(method["lle"]),
            loc=np.array(method["loc"]),
        )
