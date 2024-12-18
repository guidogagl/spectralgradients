from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from lightning.pytorch import seed_everything

import torch

from loguru import logger

from torchmetrics import Accuracy

import pandas as pd

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

        # logger.info( x.shape)

        batch_size, n = x.shape
        y = self.nn(x.reshape(batch_size, *self.input_shape))

        y = self.softmax(y).reshape(batch_size, -1)

        if not batched:
            return y.reshape(-1)

        return y


class EvaluationModule(pl.LightningModule):
    def __init__(
        self,
        nn: torch.nn.Module,
        explainers: torch.nn.ModuleList,
        explainers_names: list,
        metrics: list,
        metrics_kwargs: list,
        metrics_names: list,
    ):
        super(EvaluationModule, self).__init__()

        self.nn = nn

        self.explainers = explainers
        self.explainers_names = explainers_names

        self.metrics = metrics
        self.metrics_kwargs = metrics_kwargs
        self.metrics_names = metrics_names

        self.acc = Accuracy(task="multiclass", num_classes=5)

    def forward(self, x, mask):

        attributions = [(attribute(x), attribute) for attribute in self.explainers]

        # each metric return a vector in b * m
        results = []
        for attr, attr_call in attributions:
            results = results + [
                torch.stack(
                    [
                        metric(f=self.nn, exp=attr_call, **kwargs)(x, attr, mask)
                        for metric, kwargs in zip(self.metrics, self.metrics_kwargs)
                    ],
                    dim=0,
                )
            ]

        results = torch.stack(results, dim=0)
        return results

    def validation_step(self, batch, batch_idx):
        # Logica di training
        inputs, mask, targets = batch

        y_pred = self.nn(inputs)

        acc = self.acc(y_pred, targets)
        self.log(
            "acc",
            acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

        results = self.forward(inputs, mask)

        for i, exp_name in enumerate(self.exp_names):
            for j, metric_name in enumerate(self.metrics_names):

                result = results[i, j]
                result = torch.tensor(
                    [result[b, targets[b]] for b in range(len(result))]
                ).mean()
                self.log(
                    f"{exp_name}-{metric_name}",
                    result,
                    on_epoch=True,
                    on_step=True,
                    prog_bar=True,
                )

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, mask, targets = batch

        y_pred = self.nn(inputs)
        acc = self.acc(y_pred, targets)
        self.log(
            "acc",
            acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

        results = self.forward(inputs, mask)
        # print(results.shape )
        for i, exp_name in enumerate(self.explainers_names):
            for j, metric_name in enumerate(self.metrics_names):

                result = results[i, j].clone()
                result = torch.stack(
                    [result[b, targets[b]] for b in range(len(result))], dim=0
                ).mean()
                self.log(
                    f"{exp_name}-{metric_name}",
                    result,
                    on_epoch=True,
                    on_step=True,
                    prog_bar=True,
                )


def evaluation_script(
    nn: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    explainers: torch.nn.ModuleList,
    explainers_names: list,
    metrics: list,
    metrics_kwargs: list,
    metrics_names: list,
    checkpoint_path="output/model/eval/",
):

    seed_everything(42, workers=True)

    eval_module = EvaluationModule(
        nn=nn,
        explainers=explainers,
        explainers_names=explainers_names,
        metrics=metrics,
        metrics_names=metrics_names,
        metrics_kwargs=metrics_kwargs,
    )

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

from torch.utils.data import SubsetRandomSampler, DataLoader
import sys, os

sys.path.append("./")

from src.synt_data import SyntDataset
from src.train.model import TimeModule

from src.posthoc.egradients import ExpectedGradients
from src.posthoc.igradients import IntegratedGradients
from src.posthoc.saliency import Saliency, InputXGradient
from src.posthoc.tsgradients import TimeSpectralGradients as TSG

from src.metrics.lle import LocalLipschitzEstimate as LLE
from src.metrics.infidelity import Infidelity
from src.metrics.localization import Localization
from src.metrics.complexity import Complexity

torch.set_float32_matmul_precision('medium')

batch_size = 256
setups = [0, 1, 2]

n_points_tsg = 10
nperseg = 100
fs = 100

if __name__ == "__main__":

    for setup in setups:

        dataset = SyntDataset(setup = setup, nperseg=nperseg, return_mask=True)

        data = SyntDataset(return_mask=True)
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

        nn = Wrapper(
            nn,
            input_shape=data[0][0].shape,
        )

        baseline_size = 4 * batch_size
        baseline = torch.randperm(len(data))[:baseline_size].long()
        baseline, _, _ = data[baseline]
        

        nyquist = fs / 2
        f_res = fs / nperseg
        freqs = torch.arange(0, nyquist + f_res, f_res)
        n_points = len(freqs) * n_points_tsg

        explainers = torch.nn.ModuleList(
            [
                Saliency(f=nn),
                InputXGradient(f=nn),
                IntegratedGradients(
                    f=nn, 
                    baselines=torch.zeros(batch_size, *data[0][0].shape),
                    n_points= n_points
                ),
                ExpectedGradients(
                    f=nn, 
                    baselines=baseline,
                    n_points= n_points
                ),
                TSG( 
                    f = nn, 
                    fs = fs,
                    nperseg = nperseg,
                    n_points = n_points_tsg,
                    strategy="highest" 
                ),
                TSG(     
                    f = nn, 
                    fs = fs,
                    nperseg = nperseg,
                    n_points = n_points_tsg,
                    strategy="lowest" 
                ),
            ]
        )

        explainers_names = ["sal", "ixg", "ig", "eg", "tsgh", "tsgl"]

        metrics = [
            Localization,
            Complexity,
            Infidelity,
            #LLE,
        ]

        metrics_kwargs = [{}, {}, {}, {}]

        metrics_names = ["loc", "comp", "inf"] # , "lle"

        evaluation_script(
            nn=nn,
            loader=loader,
            explainers=explainers,
            explainers_names=explainers_names,
            metrics=metrics,
            metrics_kwargs=metrics_kwargs,
            metrics_names=metrics_names,
            checkpoint_path=f"output/model/setup{setup}/metrics/",
        )
