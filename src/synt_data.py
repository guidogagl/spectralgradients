import torch
import numpy as np

from scipy import signal

from typing import List

from scipy import signal as F

from scipy.ndimage import gaussian_filter1d

from loguru import logger

BACKGROUND_NOISE = [{"freq": 5}]

CLASS_DESC = [
    [{"freq": 45}],
    [{"freq": 15}],
    [{"freq": 25, "sec": (0, 0.5)}],
    [{"freq": 25, "sec": (0.5, 1)}],
]


class SyntDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        class_desc: dict = CLASS_DESC,  # frequencies of the sinusoidal signals for each class
        n_samples: int = 10000,  # number of samples per class
        fs: float = 100,  # sampling frequency
        length: int = 10,  # length of the time series in seconds
        bandwidth: float = 5.0,  # bandwidth of the Gaussian window
        return_mask : float = False,
    ):

        # create n_samples time series for each class
        self.data = []
        self.masks = []
        self.labels = []

        for i, freqs in enumerate(class_desc):
            try:
                with open(f"output/data/synt_class={i}.npy", "rb") as f:
                    temp = np.load(f)
                    time_series, masks, labels = (
                        temp["signal"],
                        temp["mask"],
                        temp["label"],
                    )

                time_series = torch.tensor(time_series)
                masks = torch.tensor(masks)
                labels = torch.tensor(labels)

                self.data.extend(time_series)
                self.masks.extend(masks)
                self.labels.extend(labels)

            except:
                # print the exception information
                # logger.exception(f"Error loading data/synt_class={i}.pt")
                print(f"Error loading output/data/synt_class={i}.pt")
                time_series, masks, labels = [], [], []

                for _ in range(n_samples):

                    signal, mask = gen_time_series(freqs, fs, length, bandwidth)
                    time_series.append(signal)
                    masks.append(mask)
                    labels.append(i)

                time_series = torch.stack(time_series).float()
                labels = torch.tensor(labels).long()
                masks = torch.stack(masks).long()

                print(f"Saving output/data/synt_class={i}.pt")
                print(time_series.shape, labels.shape)

                with open(f"output/data/synt_class={i}.npy", "wb") as f:
                    np.savez(
                        f,
                        signal=time_series.numpy(),
                        mask=masks.numpy(),
                        label=labels.numpy(),
                    )

                self.data.extend(time_series)
                self.labels.extend(labels)
                self.masks.extend(masks)

        # add baseline as a class
        time_series, masks, labels = [], [], []
        for _ in range(n_samples // 2):
            signal, mask = gen_time_series(None, fs, length, bandwidth)
            time_series.append(signal)
            masks.append(mask)
            labels.append(i + 1)

        for _ in range(n_samples // 2):
            signal, mask = torch.zeros(fs * length), torch.zeros(fs * length)
            time_series.append(signal)
            masks.append(mask)
            labels.append(i + 1)

        self.data.extend(time_series)
        self.masks.extend(masks)
        self.labels.extend(labels)

        self.n_class = i + 2

        self.data = torch.stack(self.data).float()
        self.masks = torch.stack(self.masks).long()
        self.labels = torch.tensor(self.labels).long()

        # compute the mean and std of the dataset
        mean = torch.mean(self.data, dim=0)
        std = torch.std(self.data, dim=0)

        # standardize the dataset
        self.data = (self.data - mean) / std

        # compute the mean power for each class
        self.mean_power = []

        for i in range(self.n_class):
            _, P = F.welch(
                self.data[self.labels == i].numpy(), fs=fs, nperseg=fs * 2, axis=1
            )
            self.mean_power.append(np.mean(P, axis=0))

        self.mean_power = np.stack(self.mean_power)
        self.mean_power = torch.tensor(self.mean_power, dtype=torch.float32)

        self.n_samples = n_samples
        self.return_mask = return_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.return_mask:
            return self.data[idx], self.masks[idx], self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]


def gen_time_series(
    desc: dict,
    fs: float,  # sampling frequency
    length: int,  # length of the time series in seconds
    bandwidth: float = 5,  # bandwidth of the Gaussian window
):

    n_samples = int(fs * length)

    # Generate white noise
    white_noise = torch.normal(mean=0, std=1, size=(n_samples,)).numpy()

    # isolate the background noise each signal shares

    # Apply bandpass filters to the white noise
    filtered_signal = np.zeros(n_samples)
    nyquist = 0.5 * fs

    for d in BACKGROUND_NOISE:
        freq = d["freq"]
        low = (freq - bandwidth / 2) / nyquist
        high = (freq + bandwidth / 2) / nyquist
        b, a = F.butter(4, [low, high], btype="bandpass")
        filt = F.filtfilt(b, a, white_noise)

        amp = np.random.uniform(0, 1)
        filtered_signal += amp * filt

    # Add the power of the class

    _mask = np.zeros(n_samples)

    if desc is not None and len(desc) > 0:
        for descriptor in desc:
            freq = descriptor["freq"]

            # amp is a random float between 0 and 1 both included
            amp = np.random.uniform(0, 1)

            low = (freq - bandwidth / 2) / nyquist
            high = (freq + bandwidth / 2) / nyquist
            b, a = F.butter(4, [low, high], btype="bandpass")
            filt = F.filtfilt(b, a, white_noise)

            # compute the lenght of the distortion
            if "sec" in descriptor:
                start, end = descriptor["sec"]

                dist_len = np.random.randint(
                    int(start * length) + 1, int(end * length)
                ) - int(start * length)
                dist_start = np.random.randint(
                    start * length, (end * length - dist_len)
                )
            else:
                dist_len = np.random.randint(1, length)
                dist_start = np.random.randint(0, length - dist_len)

            dist_start = int(dist_start * fs)
            dist_len = int(dist_len * fs)

            mask = np.zeros(n_samples)
            mask[dist_start : dist_start + dist_len] = 1
            _mask[dist_start : dist_start + dist_len] = 1

            # apply the mask to the filtered signal
            filt = filt * mask

            sigma = 2
            # apply a Gaussian filter to the signal to avoid discontinuities

            if dist_start != 0:
                start_spacing = 3 * sigma if dist_start > 3 * sigma else dist_start
                filt[dist_start - start_spacing : dist_start + (3 * sigma)] = (
                    gaussian_filter1d(
                        filt[dist_start - start_spacing : dist_start + (3 * sigma)],
                        sigma,
                    )
                )

            if dist_start + dist_len != n_samples:
                end_spacing = (
                    3 * sigma
                    if n_samples - dist_start + dist_len > 3 * sigma
                    else n_samples - dist_start + dist_len
                )
                filt[
                    dist_start
                    + dist_len
                    - (3 * sigma) : dist_start
                    + dist_len
                    + end_spacing
                ] = gaussian_filter1d(
                    filt[
                        dist_start
                        + dist_len
                        - (3 * sigma) : dist_start
                        + dist_len
                        + end_spacing
                    ],
                    sigma,
                )

            filtered_signal += amp * filt

        if not _mask.any() and desc is not None:
            raise ValueError(
                "Generated mask is entirely zeros. Please check the descriptor or parameters."
            )

    return torch.tensor(filtered_signal, dtype=torch.float32), torch.tensor(
        _mask, dtype=torch.long
    )
