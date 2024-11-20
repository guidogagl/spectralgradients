import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm


class TimeModule(pl.LightningModule):
    def __init__(self, model : str = "conv", fs : float = 100, length : int = 10, n_classes : int = 6):
        super(TimeModule, self).__init__()
        self.n_classes = n_classes + 1

        if model == "conv":
            self.nn = TimeConvNet(fs, length, self.n_classes)
        elif model == "rnn":
            self.nn = TimeRNNNet(fs, length, self.n_classes)
        else:
            raise ValueError("Model must be one of 'conv' or 'rnn'")
        
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

        self.log(f"{log}_loss", loss, prog_bar=True)
        self.log(f"{log}_acc", self.wacc(outputs, targets), prog_bar=True)
        self.log(f"{log}_f1", self.wf1(outputs, targets), prog_bar=True)

        if log_metrics:
            self.log(f"{log}_ck", self.ck(outputs, targets))
            self.log(f"{log}_pr", self.pr(outputs, targets))
            self.log(f"{log}_rc", self.rc(outputs, targets))
            self.log(f"{log}_macc", self.macc(outputs, targets))
            self.log(f"{log}_mf1", self.mf1(outputs, targets))

        return loss

    def training_step(self, batch, batch_idx):
        # get the logged metrics
        if "val_loss" not in self.trainer.logged_metrics:
            self.log("val_loss", float("inf"))

        # Logica di training
        inputs, targets = batch
        outputs = self(inputs)

        return self.compute_loss (outputs, targets)

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        outputs = self(inputs)

        return self.compute_loss (outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch

        outputs = self(inputs)

        return self.compute_loss (outputs, targets, "test", log_metrics=True)




class SimpleNet( nn.Module ):
    def __init__(self, 
            fs : float = 100, # sampling frequency
            length : int = 10, # length of the time series in seconds
            n_classes : int = 6, # number of classes
        ):
        super().__init__()
    
        
        self.fs = fs
        self.length = length
        self.n_classes = n_classes 
        
        self.input_shape = (fs * length,)
      
    def forward( self, x ):
        # x is batch_size, fs * length
        
        # first we need to compute the power spectral density of the signals

        x = torch.fft.rfft(x, n = self.fs*self.length, dim = -1)
        # x shape is batch_size, fs * length // 2 + 1 only positives frequencies
        
        # compute the power spectral density
        x = ( x.abs()**2 ) *( 1 / (self.fs * self.length) )
        # x shape is batch_size, fs * length // 2 + 1 only positives frequencies
        
        # convert the power spectral density to decibels
        x = 10 * torch.log10(x + 1e-6) 
                
        return x


class TimeConvNet( nn.Module ):
    def __init__(self, 
            fs : float = 100, # sampling frequency
            length : int = 10, # length of the time series in seconds
            n_classes : int = 6, # number of classes
        ):
        super().__init__()
        
        self.fs = fs
        self.length = length
        self.n_classes = n_classes 
        
        self.input_shape = (fs * length,)
        self.conv1 = nn.Conv1d(1, 4, kernel_size = fs, stride = fs // 2)
        self.conv2 = nn.Conv1d(4, 8, kernel_size = 4, stride = 2 )
        self.conv3 = nn.Conv1d(8, 16, kernel_size = 8, stride = 4 )        
        self.flatten = nn.Flatten()
        
        self.lin1 = nn.Linear(16 , self.n_classes)          
    def forward( self, x ):
        # x is batch_size, fs * length
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.flatten(x)
        x = self.lin1(x)
        
        return x
            

class TimeRNNNet( nn.Module ):
    def __init__(self, 
            fs : float = 100, # sampling frequency
            length : int = 10, # length of the time series in seconds
            n_classes : int = 6, # number of classes
        ):
        super().__init__()
        
        self.fs = fs
        self.length = length
        self.n_classes = n_classes 
        
        self.input_shape = (fs * length,)
        
        # first extract the encodings from the input time series
        self.conv1 = nn.Conv1d(1, 2, kernel_size = fs, stride = fs // 2)
        self.conv2 = nn.Conv1d(2, 4, kernel_size = 4, stride = 2 )
        
        self.rnn = nn.GRU(4, 8, num_layers = 2, batch_first = True, bidirectional=True)
        
        self.lin1 = nn.Linear(16 , self.n_classes)
                
    def forward( self, x ):
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