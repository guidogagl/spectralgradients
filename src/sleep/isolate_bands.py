
import numpy as np
from scipy import signal 

from physioex.preprocess.sleepedf import SLEEPEDFPreprocessor

from loguru import logger

def stopband_filter(x, fs, cutoff, filter_type="low", order=5):
    
    if filter_type == "low":
        b, a = signal.butter(order, cutoff, fs = fs, btype='lowpass')
    elif filter_type == "high":
        b, a = signal.butter(order, cutoff, fs = fs, btype='highpass')
    else:
        raise ValueError("filter_type must be 'low' or 'high'")

    return signal.filtfilt(b, a, x)


def isolate_bands(signals, preprocessor_shape, fs=100 ):

    num_windows, n_channels, n_timestamps = signals.shape

    # take only the EEG signal 
    signals = signals[:, 0]

    # now concatenate the epochs for fast filter computation
    signals = np.reshape( signals, -1)

    # isolate the 5 sleep bands from the highest to the lowest with a lowpass filter
    
    gamma = stopband_filter( signals, fs, 30 )
    beta = stopband_filter( gamma, fs, 14 )
    sigma = stopband_filter( beta, fs, 12 )
    alpha = stopband_filter( sigma, fs, 8 )
    theta = stopband_filter( alpha, fs, 4 )
    delta = np.zeros_like( signals )
    
    bands = np.stack([ signals, gamma, beta, sigma, alpha, theta, delta], axis=0)
    
    if np.sum( np.isnan( bands )) :
        logger.error( "Nan values found after filtering...")
        exit()
        
    # Reshape back to the format NUM_WINDOWS, BANDS, N_TIMESTAMPS
    bands = np.reshape(bands, (7, num_windows, n_timestamps))
    bands = np.transpose(bands, (1, 0, 2))  # Transpose to NUM_WINDOWS, BANDS, N_TIMESTAMPS
    
    return bands.astype( np.float32 )




if __name__ == "__main__":
    
    preprocessor = SLEEPEDFPreprocessor(
        preprocessors_name = [ "isolate_bands"],
        preprocessors = [ isolate_bands ],
        preprocessor_shape = [[7, 3000]], # 5 bands isolation + raw signal
        data_folder = "output/data/"
    )

    preprocessor.run()
