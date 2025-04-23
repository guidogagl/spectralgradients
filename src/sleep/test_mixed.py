import torch    
import os 

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
# plot the classification report 
from sklearn.metrics import classification_report
import pandas as pd
from loguru import logger

from tqdm import tqdm
import torch


from lightning.pytorch import seed_everything
from torch import set_float32_matmul_precision

from physioex.train.models import load_model
from physioex.train.networks import config
from physioex.data import PhysioExDataModule

from sklearn.neighbors import KNeighborsClassifier

##### PARAMETERS #####

dataset = "sleepedf"
batch_size = 256

#####

#### CONSTANTS ####

device = "cuda" if torch.cuda.is_available() else "cpu"
config = config["default"]
config["model_kwargs"]["in_channels"] = 1
config["model_kwargs"]["sequence_length"] = 21

ckpt_path = f"output/model/tinytransformer/{dataset}/checkpoint.ckpt"
basedir = os.path.dirname(ckpt_path)

if not os.path.exists(basedir):
    os.makedirs(basedir)

###################

def get_importances( model, datamodule, basedir = basedir ):
        # check if the data is already cached in memory
    if  os.path.exists(os.path.join(basedir, "x_train.npy")) and \
        os.path.exists(os.path.join(basedir, "y_train.npy")) and \
        os.path.exists(os.path.join(basedir, "y_hat_train.npy")):
        
        logger.info("Loading cached data...")
        x_train = np.load(os.path.join(basedir, "x_train.npy"))
        y_train = np.load(os.path.join(basedir, "y_train.npy"))
        y_hat_train = np.load(os.path.join(basedir, "y_hat_train.npy"))
    else:        
        logger.info( "Extracting bands importance from training set ...")
        x_train, y_train, y_hat_train = extract_bands_importance(model, datamodule.train_dataloader())
        print( "x_train shape: ", x_train.shape )
        print( "y_train shape: ", y_train.shape )
        print( "y_hat_train shape: ", y_hat_train.shape )

        # cache the values on the disk to avoid recomputing them
        np.save( os.path.join(basedir, "x_train.npy"), x_train )
        np.save( os.path.join(basedir, "y_train.npy"), y_train )
        np.save( os.path.join(basedir, "y_hat_train.npy"), y_hat_train )
        
    tmp = []
    for i in range( x_train.shape[0] ):
        tmp.append( x_train[i, :, y_train[i]] )
    x_train = np.stack( tmp, axis=0 )
    del tmp
        
    
    # check if the data is already cached in memory
    if os.path.exists(os.path.join(basedir, "x_val.npy")) and \
        os.path.exists(os.path.join(basedir, "y_val.npy")) and \
        os.path.exists(os.path.join(basedir, "y_hat_val.npy")):
        
        logger.info("Loading cached data...")
        x_val = np.load(os.path.join(basedir, "x_val.npy"))
        y_val = np.load(os.path.join(basedir, "y_val.npy"))
        y_hat_val = np.load(os.path.join(basedir, "y_hat_val.npy"))
    else:
        logger.info("Extracting bands importance from validation set ...")
        x_val, y_val, y_hat_val = extract_bands_importance(model, datamodule.val_dataloader())

        # cache the values on the disk to avoid recomputing them
        np.save( os.path.join(basedir, "x_val.npy"), x_val )
        np.save( os.path.join(basedir, "y_val.npy"), y_val )
        np.save( os.path.join(basedir, "y_hat_val.npy"), y_hat_val )

    # check if the data is already cached in memory
    if os.path.exists(os.path.join(basedir, "x_test.npy")) and \
        os.path.exists(os.path.join(basedir, "y_test.npy")) and \
        os.path.exists(os.path.join(basedir, "y_hat_test.npy")):
        
        logger.info("Loading cached data...")
        x_test = np.load(os.path.join(basedir, "x_test.npy"))
        y_test = np.load(os.path.join(basedir, "y_test.npy"))
        y_hat_test = np.load(os.path.join(basedir, "y_hat_test.npy"))
    
    else:        
        logger.info("Extracting bands importance from test set ...")
        x_test, y_test, y_hat_test = extract_bands_importance(model, datamodule.test_dataloader())    

        # cache the values on the disk to avoid recomputing them
        np.save( os.path.join(basedir, "x_test.npy"), x_test )
        np.save( os.path.join(basedir, "y_test.npy"), y_test )
        np.save( os.path.join(basedir, "y_hat_test.npy"), y_hat_test )
            
    return x_train, y_train, y_hat_train, x_val, y_val, y_hat_val, x_test, y_test, y_hat_test

def normalize( x ):
    
    # x shape: ( batch_size, 6 )
    x_shape = x.shape
    x = np.reshape( x, (-1, 6) )
    
    x = x / ( np.sum( x, axis = -1, keepdims=True ) + 1e-8 )
    
    return x.reshape( x_shape )

##### FUNCTIONS #####
@torch.no_grad()
def evaluate_model(model, dataloader, basedir = basedir):
    
    y_true, y_pred = [], []
    model = model.to(device)

    for batch in tqdm( dataloader ):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        y_true.extend(y.view(-1).cpu().numpy())
        y_pred.extend(torch.argmax(y_hat, dim=-1).view(-1).cpu().numpy())

        del x, y, y_hat
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    model = model.to("cpu")
        
    # plot the confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Wake", "NREM1", "NREM2", "NREM3", "REM"],
        cmap=plt.cm.Blues,
        normalize="true",
        include_values=True,
        values_format=".2%",
    )

    disp.ax_.set_title("Confusion Matrix")
    plt.savefig(os.path.join(basedir, "confusion_matrix.png"), dpi=300)
    plt.show()
    plt.close()
    
    # print the classification report and save it to a csv file
    #report = classification_report(y_true, y_pred, target_names=["Wake", "NREM1", "NREM2", "NREM3", "REM"])
    report = classification_report(y_true, y_pred, target_names=["Wake", "NREM1", "NREM2", "NREM3", "REM"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(basedir, "classification_report.csv"))
    print(report_df)

    return y_true, y_pred
    

def stopband_filter(x, fs, cutoff, filter_type="low", order=5):
    from scipy import signal    

    if filter_type == "low":
        b, a = signal.butter(order, cutoff, fs = fs, btype='lowpass')
    elif filter_type == "high":
        b, a = signal.butter(order, cutoff, fs = fs, btype='highpass')
    else:
        raise ValueError("filter_type must be 'low' or 'high'")

    return signal.filtfilt(b, a, x)


def isolate_bands(signals, fs=100 ):

    in_shape = signals.shape
    # in_shape : batch_size, sequence_length, channels, time_stamps
    
    if len( in_shape ) == 4: # batched data
        signals = np.reshape( signals, (signals.shape[0], -1))
    else: # unbatched data
        signals = np.reshape( signals, -1)
        
    
    gamma = stopband_filter( signals, fs, 30 )
    beta = stopband_filter( gamma, fs, 14 )
    sigma = stopband_filter( beta, fs, 12 )
    alpha = stopband_filter( sigma, fs, 8 )
    theta = stopband_filter( alpha, fs, 4 )
    delta = np.zeros_like( signals )
    
    bands = np.stack([ signals, gamma, beta, sigma, alpha, theta, delta], axis=0)
            
    # Reshape back to the format NUM_WINDOWS, BANDS, N_TIMESTAMPS
    bands = np.reshape(bands, (7, *in_shape))    
    return bands.astype( np.float32 )


@torch.no_grad()
def extract_bands_importance( model, dataloader ):
    importances, y_true, y_hats = [], [], []
    
    model = model.to(device)        

    for batch in tqdm(dataloader):
        x, y = batch

        # x shape : batch_size, sequence_lenght, channels, time_stamps
        # y shape : batch_size, sequence_lenght
        
        bands = torch.tensor( isolate_bands(x.cpu().numpy()), dtype=torch.float32 ).to( device )
        # bands shape : (7, batch_size, 21, 1, 3000)

        bands = bands.reshape( -1, 21, 1, 3000 )        
        y_hat = model( bands ).cpu().detach().numpy()
        del bands
        y = y.cpu().numpy().reshape(-1)

        y_hat = y_hat.reshape(7, -1, 5)        
        
        # calcola l'importanza per ogni banda
        importance = np.zeros( ( 6, y_hat.shape[1], 5 ), dtype=np.float32 )
        for i in range(1, y_hat.shape[0] ):
            # l'importanza della banda i-esima è calcolata come f(x_i) - f(x_{i-1})
            importance[ i - 1] = ( y_hat[ i - 1] - y_hat[i] )
            # se ci sono valori negativi, mettili a zero
            importance[ i - 1][importance[ i - 1] < 0] = 0        
        
        importance = importance.reshape(6, -1, 5)

        y_hat = y_hat[0].reshape(-1, 5)
        
        # salva le importanze
        importances.append(importance)
        y_true.append(y)
        y_hats.append(y_hat)

    model = model.to("cpu")         

    importances = np.concatenate(importances, axis=1)
    importances = np.transpose(importances, (1, 0, 2)) # batch_size, 6, 5
    y_true = np.concatenate(y_true, axis=0)
    y_hats = np.concatenate(y_hats, axis=0)
    
    print( "importances shape: ", importances.shape )
    print( "y_true shape: ", y_true.shape )
    print( "y_hats shape: ", y_hats.shape )
    
    return importances, y_true, y_hats


def predict_mixed( x_train, y_train, k, x_test, y_hat_test ):

    tmp = np.copy( x_test)
    tmp = np.transpose(tmp, (0, 2, 1) ).reshape(-1, 6)
    tmp_preds = np.copy( y_hat_test )
    
    knn = KNeighborsClassifier(n_neighbors=k).fit( x_train, y_train )

    y_imp_preds = knn.predict_proba( tmp ) # shape (batch_size * 5, 5)
    y_imp_preds = y_imp_preds.reshape(-1, 5, 5) # shape (batch_size, 5, 5)

    y_imp_preds = y_imp_preds.diagonal(axis1=1, axis2=2) # shape (batch_size, 5)
    
    # add the predictions from the model
    
    max_confidence = np.max( tmp_preds, axis = -1) # shape ( batch_size )
    print( "max_confidence shape: ", max_confidence.shape )
    # when the max confidence is less than  0.3 ( 1/5 + 0.1 treshold ) use the explanations
    #mask = max_confidence < 0.5

    #y_imp_preds = np.copy( y_imp_preds )
    #y_imp_preds[mask, :] = 0    
    #y_imp_preds = y_imp_preds + tmp_preds
    
    y_imp_preds = np.argmax(y_imp_preds, axis=1) # shape (batch_size, 5)
    
    return y_imp_preds    


ks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

def evaluate_best_k( x_train, y_train, x_test, y_test, y_hat_test ):
    best_k = 1
    best_acc = 0
    for k in tqdm(ks):
        y_imp_preds = predict_mixed( x_train, y_train, k, x_test, y_hat_test )
        acc = np.sum( y_imp_preds == y_test ) / len(y_test)

        print(f"Accuracy for k={k}: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"Best k: {best_k}, Best accuracy: {best_acc}")
    
    return best_k

#####################

if __name__ == "__main__":
    logger.info("Seeding...")
    seed_everything(42, workers=True)
    set_float32_matmul_precision("medium")

    logger.info("Loading model...")
    from src.sleep.train import TinyTransformerNet
    # Load the model

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
    
    model = TinyTransformerNet.load_from_checkpoint( ckpt_path, config = model_kwargs ).eval()
    # add a softmax layer to the model
    model = torch.nn.Sequential(
        model,
        torch.nn.Softmax(dim=-1)
    )

    model = model.to(device)
    # print model summary
    print(model)
        
    logger.info("Loading dataset...")    
    datamodule = PhysioExDataModule(
        datasets = [dataset],
        batch_size = batch_size,
        data_folder = os.environ["PHYSIOEXDATA"],
        num_workers = os.cpu_count(),
    )

    logger.info("Evaluating model on the test set...")
    evaluate_model(model, datamodule.test_dataloader())
    
    logger.info("Pruning the training set...")
    #remove the missclassification from the training set
    mask = ( y_train == np.argmax( y_hat_train, axis = -1 ) )
    x_train = x_train[mask]
    y_train = y_train[mask]
    y_hat_train = y_hat_train[mask]

    # normalize the data    
    x_train = normalize( x_train )
    x_val = normalize( x_val )
    x_test = normalize( x_test )

    #logger.info("Evaluating best k...")
    #k = evaluate_best_k( x_train, y_train, x_val, y_val, y_hat_val )
    
    k = 5
    
    logger.info("Predicting mixed on test set...")
    y_imp_preds = predict_mixed( x_train, y_train, k, x_test, y_hat_test )
    
    # plot the confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_imp_preds,
        display_labels=["Wake", "NREM1", "NREM2", "NREM3", "REM"],
        cmap=plt.cm.Blues,
        normalize="true",
        include_values=True,
        values_format=".2%",
    )

    disp.ax_.set_title("Confusion Matrix")

    plt.savefig(os.path.join(basedir, "confusion_matrix_imp.png"), dpi=300)
    plt.close()
    
    # print the classification report and save it to a csv file
    #report = classification_report(y_true, y_pred, target_names=["Wake", "NREM1", "NREM2", "NREM3", "REM"])
    report = classification_report(y_test, y_imp_preds, target_names=["Wake", "NREM1", "NREM2", "NREM3", "REM"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(basedir, "classification_report_imp.csv"))
    print(report_df)