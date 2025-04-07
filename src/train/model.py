import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm


class TimeModule(pl.LightningModule):
    def __init__(
        self,
        model: str = "conv",
        input_shape: tuple = [1, 1000],
        fs: int = 100,
        n_classes: int = 6,
    ):
        super(TimeModule, self).__init__()
        self.n_classes = n_classes

        self.nn = TimeConvNet(input_shape, fs, self.n_classes)

        self.wacc = tm.Accuracy(
            task="multiclass", num_classes=self.n_classes, average="weighted"
        )
        self.macc = tm.Accuracy(
            task="multiclass", num_classes=self.n_classes, average="macro"
        )
        self.wf1 = tm.F1Score(
            task="multiclass", num_classes=self.n_classes, average="weighted"
        )
        self.mf1 = tm.F1Score(
            task="multiclass", num_classes=self.n_classes, average="macro"
        )
        self.ck = tm.CohenKappa(task="multiclass", num_classes=self.n_classes)
        self.pr = tm.Precision(
            task="multiclass", num_classes=self.n_classes, average="weighted"
        )
        self.rc = tm.Recall(
            task="multiclass", num_classes=self.n_classes, average="weighted"
        )

        # loss
        self.loss = nn.CrossEntropyLoss()

        # learning rate
        self.learning_rate = 1e-4
        self.weight_decay = 1e-6

    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return self.opt

    def forward(self, x):
        return self.nn(x)

    def compute_loss(
        self,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):

        batch_size, n_class = outputs.size()

        loss = self.loss(outputs, targets)

        self.log(f"{log}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{log}_acc", self.wacc(outputs, targets), prog_bar=True, sync_dist=True)
        self.log(f"{log}_f1", self.wf1(outputs, targets), prog_bar=True, sync_dist=True)

        if log_metrics:
            self.log(f"{log}_ck", self.ck(outputs, targets), sync_dist=True)
            self.log(f"{log}_pr", self.pr(outputs, targets), sync_dist=True)
            self.log(f"{log}_rc", self.rc(outputs, targets), sync_dist=True)
            self.log(f"{log}_macc", self.macc(outputs, targets), sync_dist=True)
            self.log(f"{log}_mf1", self.mf1(outputs, targets), sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        # get the logged metrics
        if "val_loss" not in self.trainer.logged_metrics:
            self.log("val_loss", float("inf"))

        # Logica di training
        inputs, targets = batch
        outputs = self(inputs)

        return self.compute_loss(outputs, targets)

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        outputs = self(inputs)

        return self.compute_loss(outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch

        outputs = self(inputs)

        return self.compute_loss(outputs, targets, "test", log_metrics=True)


class TimeConvNet(nn.Module):
    def __init__(
        self,
        input_shape,
        fs: int = 100,
        n_classes: int = 6,  # number of classes
    ):
        super().__init__()

        self.n_classes = n_classes
        self.input_shape = input_shape

        self.conv = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=fs, stride=fs // 2),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Flatten(),
        )

        out_shape = self.conv(torch.rand(1, 1, *input_shape)).shape[-1]

        self.lin1 = nn.Linear(out_shape, self.n_classes)

    def forward(self, x):
        # x is batch_size, fs * length
        if len(x.shape) == 1:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        x = x.reshape(batch_size, 1, x.shape[-1])

        x = self.conv(x)
        x = self.lin1(x)

        if batch_size == 1:
            x = x.squeeze(0)
        return x


class TimeRNNNet(nn.Module):
    def __init__(
        self,
        fs: float = 100,  # sampling frequency
        length: int = 10,  # length of the time series in seconds
        n_classes: int = 6,  # number of classes
    ):
        super().__init__()

        self.fs = fs
        self.length = length
        self.n_classes = n_classes

        self.input_shape = (fs * length,)

        # first extract the encodings from the input time series
        self.conv1 = nn.Conv1d(1, 2, kernel_size=fs, stride=fs // 2)
        self.conv2 = nn.Conv1d(2, 4, kernel_size=4, stride=2)

        self.rnn = nn.GRU(4, 8, num_layers=2, batch_first=True, bidirectional=True)

        self.lin1 = nn.Linear(16, self.n_classes)

    def forward(self, x):
        # x is batch_size, fs * length
        batch_size = x.shape[0]
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)

        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)

        x = x.sum(1)
        x = self.lin1(x)
        return x
