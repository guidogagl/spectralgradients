#!/usr/bin/env python3
"""Compute freq/time localization from saved attributions for synthetic setups.

Loads saved attribution .pt files + dataset masks, computes localization metrics,
and saves results to JSON alongside existing metrics.

Usage::
    python synt/compute_localization.py
"""

import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
from physioex.explain.posthoc.metrics import localization
from synt.data import SyntDataset, SETUPS

SETUPS_LIST = ["synt-setup0", "synt-setup1", "synt-setup2"]
RESULTS_ROOT = Path(os.environ.get("SG_RESULTS", "results"))
DATA_ROOT = Path(os.environ.get("SG_DATA", Path(__file__).resolve().parents[1] / "data")) / "synt"
EXPLAINERS = ["SG", "IG-time", "IG-bal", "IG-freq",
              "Sal-time", "Sal-bal", "Sal-freq",
              "IxG-time", "IxG-bal", "IxG-freq"]

FS = 100.0
LENGTH = 10
BANDWIDTH = 5.0
N = int(FS * LENGTH)  # 1000


def build_freq_mask(n, y, setup_desc):
    """Build frequency-domain binary mask for class y."""
    freqs = torch.fft.rfftfreq(n, d=1.0 / FS)
    mask = torch.zeros_like(freqs, dtype=torch.bool)
    if y < len(setup_desc):
        freq_0 = setup_desc[y][0]["freq"]
        low = max(freq_0 - BANDWIDTH, 0)
        high = min(freq_0 + BANDWIDTH, FS / 2)
        mask[(freqs >= low) & (freqs <= high)] = True
    return mask


def interpolate_1d(x, target_len):
    """Interpolate 1D tensor to target length."""
    if x.shape[-1] == target_len:
        return x
    return torch.nn.functional.interpolate(
        x.unsqueeze(0).float(), size=target_len, mode="linear", align_corners=False
    ).squeeze(0)


def main(results_root=RESULTS_ROOT, data_root=DATA_ROOT, setups=None):
    for setup_idx, setup_name in enumerate(SETUPS_LIST):
        if setups is not None and setup_idx not in setups:
            continue
        print(f"\n{'='*60}")
        print(f"  {setup_name}")
        print(f"{'='*60}")

        setup_desc = SETUPS[setup_idx]["desc"]
        result_dir = Path(results_root) / setup_name

        # Load dataset with masks
        ds = SyntDataset(
            setup=setup_idx, n_samples=1000, fs=FS, length=LENGTH,
            bandwidth=BANDWIDTH, return_mask=True, output_dir=str(data_root)
        )

        # Load SG attributions to get labels and sample count
        sg_attr_file = result_dir / "SG_attributions.pt"
        sg_data = torch.load(sg_attr_file, map_location="cpu", weights_only=False)
        labels = sg_data["labels"]  # (N_samples,)
        n_samples = len(labels)

        # The benchmark persisted the exact sample set it explained (same order as the
        # attribution tensors): match each cached signal to its dataset row to fetch the masks.
        caches = sorted(result_dir.glob("samples_n*_s*.npz"))
        if not caches:
            raise FileNotFoundError(f"no samples_n*_s*.npz cache in {result_dir}; run synt/benchmark.py first")
        cache = np.load(caches[-1])
        if "indices" in cache.files:
            sample_indices = [int(i) for i in cache["indices"][:n_samples]]
        else:   # older cache without row indices: match the standardised signals byte for byte
            cached = cache["data"]
            row_of = {row.numpy().tobytes(): i for i, row in enumerate(ds.data)}
            sample_indices = []
            for k in range(min(n_samples, len(cached))):
                key = np.asarray(cached[k], dtype=str(ds.data.dtype).replace("torch.", "")).tobytes()
                if key not in row_of:
                    raise KeyError(f"cached sample {k} not found in the dataset under {data_root} "
                                   "(cache without 'indices'; rerun synt/benchmark.py)")
                sample_indices.append(row_of[key])

        # Build masks for each sample
        time_masks = []
        freq_masks = []
        signals = []
        for idx in sample_indices:
            _, tmask, label = ds[idx]
            y = int(label)
            fmask = build_freq_mask(N, y, setup_desc)
            time_masks.append(tmask)
            freq_masks.append(fmask)
            signals.append(ds.data[idx])

        # Common interpolation size (max freq bins across all explainers)
        common_freq = N // 2 + 1  # 501 DFT bins
        common_time = N  # 1000 time points

        for ename in EXPLAINERS:
            attr_file = result_dir / f"{ename}_attributions.pt"
            if not attr_file.exists():
                print(f"  {ename}: attr file not found, skipping")
                continue

            data = torch.load(attr_file, map_location="cpu", weights_only=False)
            attrs = data["attributions"]  # (N, F, T)

            freq_loc_vals = []
            time_loc_vals = []

            for si in range(min(n_samples, attrs.shape[0])):
                y = int(labels[si])
                if y >= len(setup_desc):
                    continue

                attr_2d = attrs[si]  # (F, T)
                n_freq_attr = attr_2d.shape[0]

                # Marginals
                freq_attr = attr_2d.sum(dim=-1)  # (F,)
                time_attr = attr_2d.sum(dim=-2)  # (T,)

                # Build freq mask in the explainer's native frequency space
                if y < len(setup_desc):
                    freq_0 = setup_desc[y][0]["freq"]
                    low = max(freq_0 - BANDWIDTH, 0)
                    high = min(freq_0 + BANDWIDTH, FS / 2)

                    if ename == "SG":
                        # SG bands: band i covers [i*freq_step, (i+1)*freq_step] Hz
                        freq_step = data["config"]["freq_step"]
                        fmask_native = torch.zeros(n_freq_attr, dtype=torch.bool)
                        for bi in range(n_freq_attr):
                            band_lo = bi * freq_step
                            band_hi = (bi + 1) * freq_step
                            if band_hi > low and band_lo < high:
                                fmask_native[bi] = True
                    else:
                        # STFT: freq bins from rfftfreq based on n_fft
                        window_cfg = data.get("window", {})
                        n_fft = window_cfg.get("n_fft", n_freq_attr * 2 - 2)
                        stft_freqs = torch.fft.rfftfreq(n_fft, d=1.0 / FS)
                        fmask_native = (stft_freqs >= low) & (stft_freqs <= high)
                        # Truncate/pad to match attr size
                        if len(fmask_native) > n_freq_attr:
                            fmask_native = fmask_native[:n_freq_attr]
                        elif len(fmask_native) < n_freq_attr:
                            pad = torch.zeros(n_freq_attr - len(fmask_native), dtype=torch.bool)
                            fmask_native = torch.cat([fmask_native, pad])
                else:
                    fmask_native = torch.zeros(n_freq_attr, dtype=torch.bool)

                # Time mask: interpolate if needed
                tmask = time_masks[si]
                time_attr_c = interpolate_1d(time_attr.unsqueeze(0), common_time) if time_attr.shape[-1] != common_time else time_attr.unsqueeze(0)

                # Compute localization in native space
                fl = localization(
                    attr=freq_attr.unsqueeze(0),
                    mask=fmask_native.unsqueeze(0).float()
                ).item()

                tl = localization(
                    attr=time_attr_c,
                    mask=tmask.unsqueeze(0).float(),
                    sign_weight=signals[si].unsqueeze(0)
                ).item()

                freq_loc_vals.append(fl)
                time_loc_vals.append(tl)

            freq_med = np.median(freq_loc_vals) if freq_loc_vals else float("nan")
            time_med = np.median(time_loc_vals) if time_loc_vals else float("nan")

            print(f"  {ename:<12} freq_loc={freq_med:.4f}  time_loc={time_med:.4f}  (n={len(freq_loc_vals)})")

            # Update the result JSON
            for json_pattern in [f"{ename}.json", f"{ename}_results.json"]:
                json_file = result_dir / json_pattern
                if json_file.exists():
                    d = json.load(open(json_file))
                    metrics = d.get("metrics", d.get("results", {}))
                    metrics["freq_localization"] = freq_loc_vals
                    metrics["time_localization"] = time_loc_vals
                    with open(json_file, "w") as f:
                        json.dump(d, f, indent=2)
                    break
            else:
                # SG might have different name
                if ename == "SG":
                    for sg_json in sorted(result_dir.glob("SG*.json")):
                        if "comparison" in sg_json.name or "summary" in sg_json.name:
                            continue
                        d = json.load(open(sg_json))
                        metrics = d.get("metrics", d.get("results", {}))
                        metrics["freq_localization"] = freq_loc_vals
                        metrics["time_localization"] = time_loc_vals
                        with open(sg_json, "w") as f:
                            json.dump(d, f, indent=2)
                        break

    print("\nDone! Localization metrics added to result JSONs.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add frequency/time localisation metrics to the synthetic result JSONs")
    ap.add_argument("--results", default=str(RESULTS_ROOT), help="results root holding synt-setup*/ (default: $SG_RESULTS or results)")
    ap.add_argument("--data-root", default=str(DATA_ROOT), help="directory with the generated synthetic datasets")
    ap.add_argument("--setups", type=int, nargs="*", default=None, help="subset of setups (0 1 2)")
    a = ap.parse_args()
    main(Path(a.results), Path(a.data_root), a.setups)
