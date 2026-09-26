import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import List

import argparse
import os

from physioex.explain.posthoc.filters import lowpass_filter, highpass_filter

BACKGROUND_NOISE = None
CLASS_DESC = None

SETUPS =[
    {
        "noise" : [{"freq": 5}],
        "desc": [
            [{"freq": 45}],
            [{"freq": 15}],
            [{"freq": 25, "sec": (0, 0.5)}],
            [{"freq": 25, "sec": (0.5, 1)}],
        ]
    },
    {
        "noise" : [{"freq": 45}],
        "desc": [
            [{"freq": 5}],
            [{"freq": 15}],
            [{"freq": 25, "sec": (0, 0.5)}],
            [{"freq": 25, "sec": (0.5, 1)}],
        ]
    },
    {
        "noise" : [
            [{"freq": 45}],
            [{"freq": 5}],
            [{"freq": 45}, {"freq": 5}]
        ],
        "desc": [
            [{"freq": 15}],
            [{"freq": 25}],
            [{"freq": 35, "sec": (0, 0.5)}],
            [{"freq": 35, "sec": (0.5, 1)}],
        ]
    },
]


class SyntDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        setup : int = 0,
        n_samples: int = 10000,  # number of samples per class
        fs: float = 100,  # sampling frequency
        length: int = 10,  # length of the time series in seconds
        bandwidth: float = 5.0,  # bandwidth of the Gaussian window
        return_mask: float = False,
        nperseg : int = 200,
        output_dir: str = "output/data",
    ):
        global BACKGROUND_NOISE, CLASS_DESC

        class_desc = SETUPS[setup]["desc"]
        noise = SETUPS[setup]["noise"]
        CLASS_DESC = class_desc
        BACKGROUND_NOISE = noise.copy()

        # create n_samples time series for each class
        self.data = []
        self.masks = []
        self.labels = []

        os.makedirs( f"{output_dir}/setup{setup}", exist_ok=True)

        for i, freqs in enumerate(class_desc):
            try:
                with open(f"{output_dir}/setup{setup}/synt_class={i}.npy", "rb") as f:
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
                # print(f"[Warn] Failed loading {output_dir}/setup{setup}/synt_class={i}.npy")
                print(f"[Warn] Error loading {output_dir}/setup{setup}/synt_class={i}.npy")
                time_series, masks, labels = [], [], []

                for _ in range(n_samples):
                    if len( noise ) > 1:
                        BACKGROUND_NOISE = noise[ _ % len(noise)].copy()
                    signal, mask = gen_time_series(freqs, fs, length, bandwidth)
                    time_series.append(signal)
                    masks.append(mask)
                    labels.append(i)

                time_series = torch.stack(time_series).float()
                labels = torch.tensor(labels).long()
                masks = torch.stack(masks).long()

                print(f"[Info] Saving {output_dir}/setup{setup}/synt_class={i}.npy")
                os.makedirs(f"{output_dir}/setup{setup}/", exist_ok= True)
                with open( f"{output_dir}/setup{setup}/synt_class={i}.npy", "wb") as f:
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
            if len( noise ) > 1:
                BACKGROUND_NOISE = noise[ _ % len(noise)].copy()

            signal, mask = gen_time_series(None, fs, length, bandwidth)
            time_series.append(signal)
            masks.append(mask)
            labels.append(i + 1)

        for _ in range(n_samples // 2):
            signal, mask = torch.zeros(int(fs * length)), torch.zeros(int(fs * length))
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
            class_data = self.data[self.labels == i]
            if class_data.numel() == 0:
                self.mean_power.append(torch.zeros(nperseg // 2 + 1))
                continue

            n = class_data.size(1)
            X = torch.fft.rfft(class_data, dim=1)
            psd = (X.abs() ** 2) / n
            self.mean_power.append(psd.mean(dim=0))

        self.mean_power = torch.stack(self.mean_power).to(dtype=torch.float32)

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
    white_noise = torch.normal(mean=0, std=1, size=(n_samples,))

    # isolate the background noise each signal shares

    # Apply bandpass filters to the white noise
    filtered_signal = torch.zeros(n_samples, dtype=torch.float32)

    for d in BACKGROUND_NOISE:
        freq = d["freq"]
        low_cut = max(freq - bandwidth / 2, 0.1)
        high_cut = min(freq + bandwidth / 2, fs / 2 - 0.1)
        filt = highpass_filter(white_noise, cutoff=low_cut, fs=fs, order=5)
        filt = lowpass_filter(filt, cutoff=high_cut, fs=fs, order=5)

        #amp = np.random.uniform(0, 1)
        filtered_signal +=  filt #* amp

    # Add the power of the class

    _mask = torch.zeros(n_samples, dtype=torch.float32)

    if desc is not None and len(desc) > 0:
        for descriptor in desc:
            freq = descriptor["freq"]

            # amp is a random float between 0 and 1 both included
            #amp = np.random.uniform(0, 1)

            low_cut = max(freq - bandwidth / 2, 0.1)
            high_cut = min(freq + bandwidth / 2, fs / 2 - 0.1)
            filt = highpass_filter(white_noise, cutoff=low_cut, fs=fs, order=5)
            filt = lowpass_filter(filt, cutoff=high_cut, fs=fs, order=5)

            # compute the lenght of the distortion
            if "sec" in descriptor:
                start, end = descriptor["sec"]

                dist_len = torch.randint(
                    int(start * length) + 1, int(end * length), (1,)
                ).item() - int(start * length)
                dist_start = torch.randint(
                    int(start * length), int(end * length - dist_len), (1,)
                ).item()
            else:
                dist_len = torch.randint(1, length, (1,)).item()
                dist_start = torch.randint(0, length - dist_len, (1,)).item()

            dist_start = int(dist_start * fs)
            dist_len = int(dist_len * fs)

            mask = torch.zeros(n_samples, dtype=torch.float32)
            mask[dist_start : dist_start + dist_len] = 1
            _mask[dist_start : dist_start + dist_len] = 1

            # apply the mask to the filtered signal
            filt = filt * mask

            sigma = 2
            # apply a Gaussian filter to the signal to avoid discontinuities

            if dist_start != 0:
                start_spacing = 3 * sigma if dist_start > 3 * sigma else dist_start
                filt[dist_start - start_spacing : dist_start + (3 * sigma)] = _gaussian_smooth(
                    filt[dist_start - start_spacing : dist_start + (3 * sigma)],
                    sigma,
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
                ] = _gaussian_smooth(
                    filt[
                        dist_start
                        + dist_len
                        - (3 * sigma) : dist_start
                        + dist_len
                        + end_spacing
                    ],
                    sigma,
                )

            filtered_signal += filt #* amp 

        if not _mask.any() and desc is not None:
            raise ValueError(
                "Generated mask is entirely zeros. Please check the descriptor or parameters."
            )

    return filtered_signal.to(torch.float32), _mask.to(torch.long)


def _gaussian_smooth(x: torch.Tensor, sigma: int) -> torch.Tensor:
    if sigma <= 0 or x.numel() == 0:
        return x
    radius = int(3 * sigma)
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)

    x1 = x.view(1, 1, -1)
    x_pad = torch.nn.functional.pad(x1, (radius, radius), mode="reflect")
    y = torch.nn.functional.conv1d(x_pad, kernel)
    return y.view(-1)


def main():
    ap = argparse.ArgumentParser(description="Generate the three synthetic benchmark datasets (one .npy per class and setup)")
    ap.add_argument("--n-samples", type=int, default=1000, help="samples per class (the paper's benchmark sets use 1000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synt"))
    ap.add_argument("--plots", action="store_true", help="also save one example plot per class")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    fs, length, bandwidth, nperseg = 100, 10, 5.0, 200
    for setup in range(len(SETUPS)):
        print(f"[Info] Generating synthetic dataset setup={setup} -> {args.out}")
        dataset = SyntDataset(
            setup=setup,
            n_samples=args.n_samples,
            fs=fs,
            length=length,
            bandwidth=bandwidth,
            return_mask=True,
            nperseg=nperseg,
            output_dir=args.out,
        )
        if not args.plots:
            print(f"[Info] Done setup={setup}")
            continue
        plot_dir = os.path.join(args.out, f"setup{setup}", "plot")
        os.makedirs(plot_dir, exist_ok=True)
        for cls in range(dataset.n_class):
            class_idx = torch.where(dataset.labels == cls)[0]
            if class_idx.numel() == 0:
                print(f"[Warn] No samples for setup {setup} class {cls}")
                continue
            idx = class_idx[torch.randint(0, class_idx.numel(), (1,)).item()].item()
            signal, mask, label = dataset[idx]
            t = torch.arange(signal.numel()) / fs
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.plot(t.numpy(), signal.numpy(), label="signal")
            ax.set_title(f"Setup {setup} | class {label}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"sample_class_{cls}.png"), dpi=150)
            plt.close(fig)
        print(f"[Info] Done setup={setup}")


if __name__ == "__main__":
    main()
