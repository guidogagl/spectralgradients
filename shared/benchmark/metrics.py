"""Shared metric computation utilities for benchmarking across domains.

Provides compute_all_metrics() function used by synt, audio, arrhythmia, and sleep benchmarks.
"""

import torch
import torch.nn.functional as F
from physioex.explain.posthoc.metrics import complexity, infidelity, tf_concentration


def _interp1d(t: torch.Tensor, size: int) -> torch.Tensor:
    """Interpolate tensor to target size.

    Args:
        t: Input tensor (B, ...) or (...)
        size: Target size

    Returns:
        Interpolated tensor
    """
    if t.dim() == 1:
        return F.interpolate(t.unsqueeze(0).unsqueeze(0), size=size, mode="linear", align_corners=False).squeeze()
    elif t.dim() == 2:
        return F.interpolate(t.unsqueeze(1), size=size, mode="linear", align_corners=False).squeeze(1)
    else:
        raise ValueError(f"Unsupported tensor dimensions: {t.dim()}")


def compute_all_metrics(
    attr: torch.Tensor,
    xb: torch.Tensor,
    f_target: callable,
    fs: float,
    band_freqs: torch.Tensor = None,
    total_len: int = None,
    max_stft_freq: int = None,
    config: dict = None
) -> dict:
    """Compute all 5 metrics for an attribution.

    Args:
        attr: Attribution tensor (n_bands, total_len) or (B, n_bands, total_len)
        xb: Input tensor (1, total_len) or (total_len,)
        f_target: Target function f(x)[:, target_class]
        fs: Sampling frequency
        band_freqs: Band frequency tensor (for SG)
        total_len: Total length of signal
        max_stft_freq: Maximum STFT frequency size (for interpolation)
        config: Metrics configuration dict (optional, uses defaults if None)

    Returns:
        Dict with metric values:
        - freq_infidelity, freq_complexity
        - time_infidelity, time_complexity
        - tfc_tf_spread
    """
    if config is None:
        config = {}

    freq_patch_size = config.get("freq_patch_size", 1)
    time_patch_size = config.get("time_patch_size", 50)

    # Squeeze batch dimension if present
    if attr.dim() == 3:
        attr = attr.squeeze(0)
    if xb.dim() == 1:
        xb = xb.unsqueeze(0)

    # Compute frequency and time attributions
    freq_attr = attr.sum(dim=-1)  # (n_bands,)
    time_attr = attr.sum(dim=-2)  # (total_len,)

    # Compute TF concentration metric
    if band_freqs is not None:
        # SG: use provided band frequencies
        tfc_r = tf_concentration(attr, fs, band_freqs.to(attr.device))
    else:
        # STFT: compute band frequencies from n_fft
        # Assume STFT with default n_fft if not provided
        n_fft = 512  # Default, will be overridden by caller
        band_freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs)
        fs_tfc = attr.shape[-1] / (total_len / fs) if total_len else fs
        tfc_r = tf_concentration(attr, fs_tfc, band_freqs.to(attr.device))

    results = {
        "tfc_tf_spread": tfc_r["tf_spread"].item()
    }

    # Interpolate freq attribution if needed
    if max_stft_freq and freq_attr.shape[-1] != max_stft_freq:
        freq_attr = _interp1d(freq_attr.unsqueeze(0), max_stft_freq).squeeze(0)

    # Interpolate time attribution if needed
    if total_len and time_attr.shape[-1] != total_len:
        time_attr = _interp1d(time_attr.unsqueeze(0), total_len).squeeze(0)

    # Ensure proper shape for metrics: add batch dimension if 1D
    if freq_attr.dim() == 1:
        freq_attr = freq_attr.unsqueeze(0)  # (1, n_bands)
    if time_attr.dim() == 1:
        time_attr = time_attr.unsqueeze(0)  # (1, total_len)

    # Frequency metrics
    results["freq_infidelity"] = infidelity(
        f_target, xb, freq_attr, domain="frequency", fs=fs, patch_size=freq_patch_size
    ).item()
    results["freq_complexity"] = complexity(freq_attr).item()

    # Time metrics
    results["time_infidelity"] = infidelity(
        f_target, xb, time_attr, domain="time", patch_size=time_patch_size
    ).item()
    results["time_complexity"] = complexity(time_attr).item()

    return results
