# Spectral Gradients: benchmark code and per-sample results

Code and results of the paper **"Spectral Gradients: Disentangled Time-Frequency Attributions to
Explain Neural Networks for Time Series"** (G. Gagliardi, A. L. Alfeo, M. G. C. A. Cimino, W. Samek,
M. De Vos; submitted to *Pattern Recognition*).

Spectral Gradients (SG) is a post-hoc attribution method that gives a joint time-frequency
explanation of a time-series classifier without a windowed transform: frequency bands are removed
along a DFT path and the importance of each band is distributed over time by path-integrated
gradients, so the frequency resolution (Δf) and the time resolution (1/fs) are chosen independently.
**The method is implemented in the [PhysioEx](https://github.com/guidogagl/physioex) library**
(`physioex.explain.posthoc.spectralgradients.SpectralGradients`, PhysioEx ≥ 2.0.1). This repository
contains what is needed to reproduce the evaluation of the paper against nine STFT-based explainers
(IG, Saliency, Input×Gradient with time-, balanced- and frequency-optimised windows) on seven case
studies: three synthetic setups, ECG beat classification (MIT-BIH), speech commands and sleep
staging (MASS, Sleep-EDF).

## Quick start

```bash
./run.sh setup                       # .venv with physioex, torch, ... ; checks the shipped classifiers
./run.sh results && ./run.sh tables  # Tables 3-6 of the paper from the archived per-sample results (no GPU)
./run.sh all --quick                 # data + small benchmark (~1 GPU-hour) + tables + figures
./run.sh benchmark --full --yes      # the paper's runs (~590 GPU-hours, see below)
```

`./run.sh` without arguments prints every stage, option and environment variable. `--dry-run`
prints the commands of a stage without running them. Requirements: Python >= 3.11, bash, `curl`
(`wget` for the Sleep-EDF mirror), a CUDA GPU for `benchmark`, `train` and Fig. 2.

| Stage | What it does |
|---|---|
| `setup` | creates `.venv`, installs `requirements.txt`, imports the PhysioEx APIs used here, verifies `models/SHA256SUMS` |
| `data` | synthetic data (`synt/data.py`, seed 42, 1.4 GB), MIT-BIH (`wfdb`, 107 MB), Speech Commands v0.02 (2.4 GB archive), Sleep-EDF sleep-cassette EDFs (7 GB, into `$PHYSIOEX_DATA/physionet-sleep-data/`), pretrained sleep models from Hugging Face (`4rooms/physioex`) |
| `train` | optional: retrains the three Conv1D classifiers into `models-retrained/` (the shipped weights in `models/` are never overwritten) |
| `benchmark` | explains the test samples of every model with SG and the nine STFT explainers; `--quick` uses 200/200/20/50 samples per domain, `--full` the whole test sets; `--domains`, `--gpu`, `--chunk-idx/--chunk-n` split the work |
| `merge` | merges chunked result files |
| `results` | downloads the per-sample results of the paper (75 JSON files, 101 MB) into `results/` |
| `tables` | `tables/comparison.tex` (Table 3), `localisation.tex` (Table 4), `effect_sizes.tex` (Table 5), `dftest.tex` (Table 6) and `comparison.csv` from `results/` |
| `figures` | Fig. 1 (method schematic), Fig. 3 (permutation convergence, from `figures/k_convergence.json`) and, with a GPU and Sleep-EDF, Fig. 2 (`sleep/plot_sleep_epoch_bands.py --freq-step 0.5 --epoch-idx 44666 --chunk 4`) |

Every stage is idempotent: an output that already exists is skipped unless `--force` is given.
`--quick` writes to `results-quick/` so that it never mixes with the paper-scale results.

## Layout

| Path | Content |
|---|---|
| `run.sh` | single entry point (stages above) |
| `configs/<domain>/<model>_{sg,stft}.yaml` | benchmark configuration per model: sampling rate, signal length, SG band width Δf and path, STFT windows (`_sg_df5`, `_sg_df2`: the coarser bands of Table 6) |
| `models/` | the five pretrained Conv1D classifiers explained in the paper (10 MB, see `models/README.md`) |
| `synt/` | synthetic setups with ground-truth time-frequency masks: `data.py`, `train.py`, `benchmark.py`, `compute_localization.py`, `k_convergence.py` |
| `arrhythmia/` | MIT-BIH beat classification: `data.py` (download and preprocessing), `train.py`, `benchmark.py` |
| `audio/` | Speech Commands v0.02: `train.py` (download and training), `benchmark.py` |
| `sleep/` | sleep staging with the PhysioEx pretrained models Chambon2018 (MASS) and TsinalisCNN (Sleep-EDF): `benchmark.py`, `plot_sleep_epoch_bands.py`, `generate_qualitative_figures.py` |
| `shared/` | explainer wrappers and metrics (`benchmark/`), table generators (`generate_comparison_table.py`, `effect_sizes.py`, `dftest_compare.py`), `merge_chunks.py`, Fig. 1 (`fig1_schema.py`) |
| `tables/` | the tables of the paper, as regenerated from the archived results |
| `results/`, `data/`, `figures/*.pdf` | outputs (not tracked) |

The name of a model (`synt-setup0`, `synt-setup1`, `synt-setup2`, `arrhythmia-cnn`, `audio-wavcnn`,
`chambon2018`, `tsinalis-2016`) is the key that links its config, its checkpoint and its result
directory `results/<model>/`.

## Data

The datasets are not redistributed; `run.sh data` fetches the open ones.

| Case study | Dataset | Source | Size |
|---|---|---|---|
| Synt-0/1/2 | synthetic, generated by `synt/data.py` (4 classes, 1000 samples per class and setup) | this repository | 1.4 GB |
| ECG | MIT-BIH Arrhythmia Database | PhysioNet, doi:10.13026/C2F305 (ODC-By 1.0) | 107 MB |
| Speech | Google Speech Commands v0.02 | `torchaudio.datasets.SPEECHCOMMANDS` (CC BY 4.0) | 2.4 GB archive, 5.5 GB on disk |
| Sleep-EDF | Sleep-EDF Database Expanded, sleep-cassette | PhysioNet, doi:10.13026/C2X676 | 7 GB (+ ~30 GB cache) |
| Sleep-MASS | Montreal Archive of Sleep Studies, SS3 (62 recordings) | MASS consortium, data-use agreement | not downloadable |

The sleep domain needs `PHYSIOEX_DATA` set to a directory that holds `physionet-sleep-data/*.edf`
(Sleep-EDF) and, if available, `MASS/Original/SS03/` (MASS). PhysioEx preprocesses the recordings
lazily into `PHYSIOEX_CACHE_DIR` (default `~/.cache/physioex`) on first use. Without MASS, the
Sleep-MASS rows are taken from the archived results (`run.sh results`).

## Compute

The paper's runs (`--full`) took about 590 GPU-hours on one NVIDIA A100 64 GB: the three synthetic
setups 3 h each, ECG 41 h, Speech 211 h (SG alone 103 h), Sleep-MASS 69 h and Sleep-EDF 259 h (SG
234 h). `run.sh benchmark --full` prints this estimate and asks for confirmation; `--domains` and
`--chunk-idx/--chunk-n` spread a domain over several GPUs, `run.sh merge` joins the chunks. `--quick`
runs the whole pipeline in about one GPU-hour on small subsets.

## Reproducibility

- `run.sh results && run.sh tables` regenerates the four tables of the paper byte for byte from the
  archived per-sample metric values (the JSON files under `results/<model>/` hold, per explainer, one
  value per test sample and metric; `results/dftest/` holds the coarser-band runs of Table 6).
- A `--full` benchmark with the shipped classifiers reproduces those values up to the numerical
  nondeterminism of GPU kernels and of the random permutations of the Shapley path (seed 42 is used
  throughout). Retrained classifiers (`run.sh train`, `SG_MODELS=models-retrained`) and regenerated
  synthetic data are statistically equivalent to the paper's, not bit-identical.
- The archived results were produced by the benchmark scripts of this repository before they were
  gathered under `run.sh`; the packaging was checked without re-running the benchmarks.

## Citation

Repository: https://github.com/guidogagl/spectralgradients (tag `v1.0.0`). Archive of the code and of
the per-sample results: Zenodo doi:10.5281/zenodo.22964216. See `CITATION.cff`.

## License

MIT (see `LICENSE`).
