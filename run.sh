#!/usr/bin/env bash
# run.sh -- reproduce the Spectral Gradients benchmark from one entry point.
#
#   ./run.sh <stage> [options]
#
#   stages   setup      create .venv and install the dependencies (physioex >= 2.0.1, torch, ...)
#            data       generate/download the datasets (synthetic, MIT-BIH, Speech Commands, Sleep-EDF)
#            train      (optional) retrain the three classifiers into models-retrained/
#            benchmark  explain the test samples with SG and the nine STFT explainers (--quick | --full)
#            merge      merge chunked result files (--chunk-idx/--chunk-n runs)
#            results    download the per-sample results of the paper (no GPU needed)
#            tables     Tables 3-6 of the paper from the results (comparison, localisation, effect sizes, band width)
#            figures    Fig. 1 (schematic), Fig. 3 (permutation convergence) and, with a GPU + Sleep-EDF, Fig. 2
#            all        setup -> data -> benchmark -> tables -> figures
#
#   options  --domains synt,arrhythmia,audio,sleep   subset of domains (default: all)
#            --quick | --full                        benchmark size: small subsets (~1 GPU-hour) or paper scale (~590 GPU-hours)
#            --yes                                   do not ask for confirmation before a --full benchmark
#            --force                                 recompute outputs that already exist
#            --gpu N                                 CUDA device index (default 0)
#            --chunk-idx i --chunk-n N               run one strided chunk of the sample set (merge with `run.sh merge`)
#            --no-gpu                                skip the steps that need a GPU (figures)
#            --dry-run                               print the commands without running them
#
#   environment
#            PHYSIOEX_DATA      directory with the raw sleep datasets: <dir>/physionet-sleep-data/*.edf (Sleep-EDF),
#                               <dir>/MASS/Original/SS03/ (MASS, under data-use agreement). Required for the sleep domain.
#            PHYSIOEX_CACHE_DIR cache of the lazily preprocessed sleep epochs (default ~/.cache/physioex)
#            CUDA_VISIBLE_DEVICES, HF_HOME, HF_HUB_OFFLINE   passed through to PyTorch / Hugging Face
#            SG_VENV SG_RESULTS SG_TABLES SG_MODELS SG_DATA  override .venv, results/, tables/, models/, data/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH=.

# ---------------------------------------------------------------- constants
ALL_DOMAINS="synt arrhythmia audio sleep"
models_of() { case $1 in synt) echo "synt-setup0 synt-setup1 synt-setup2";; arrhythmia) echo arrhythmia-cnn;; audio) echo audio-wavcnn;; sleep) echo "tsinalis-2016 chambon2018";; *) echo "unknown domain: $1" >&2; exit 2;; esac; }
quick_n()   { case $1 in synt|arrhythmia) echo 200;; audio) echo 20;; sleep) echo 50;; esac; }   # samples per model with --quick
full_h()    { case $1 in synt-*) echo 3.3;; arrhythmia-cnn) echo 41;; audio-wavcnn) echo 211;; chambon2018) echo 69;; tsinalis-2016) echo 259;; esac; }  # GPU-hours of the paper's runs (one A100 64 GB)
EXPLAINERS="SG IG-time IG-bal IG-freq Sal-time Sal-bal Sal-freq IxG-time IxG-bal IxG-freq"
RESULTS_URL_GH="https://github.com/guidogagl/spectralgradients/releases/download/v1.0.0/spectralgradients-per-sample-results.tar.gz"
RESULTS_URL_ZENODO="https://zenodo.org/records/22964216/files/spectralgradients-per-sample-results.tar.gz?download=1"
RESULTS_SHA256="1798276f12eb629fcf87e7ea941197ff9a0deaa567feb244e55b4dd3a43414e2"
VENV=${SG_VENV:-.venv}; MODELS_DIR=${SG_MODELS:-models}; DATA=${SG_DATA:-data}

# ---------------------------------------------------------------- options
STAGE=${1:-}; [ -n "$STAGE" ] || { sed -n '2,32p' "$0"; exit 2; }; shift
DOMAINS=$ALL_DOMAINS MODE="" YES=0 FORCE=0 GPU=0 CHUNK_IDX="" CHUNK_N="" NOGPU=0 DRY=0
while [ $# -gt 0 ]; do case $1 in
  --domains) DOMAINS=${2//,/ }; shift;; --quick) MODE=quick;; --full) MODE=full;; --yes) YES=1;; --force) FORCE=1;;
  --gpu) GPU=$2; shift;; --chunk-idx) CHUNK_IDX=$2; shift;; --chunk-n) CHUNK_N=$2; shift;; --no-gpu) NOGPU=1;; --dry-run) DRY=1;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; shift; done
[ "$STAGE" = all ] && [ -z "$MODE" ] && MODE=quick
if [ "$MODE" = quick ]; then RESULTS=${SG_RESULTS:-results-quick}; TABLES=${SG_TABLES:-results-quick/tables}
else RESULTS=${SG_RESULTS:-results}; TABLES=${SG_TABLES:-tables}; fi
export SG_RESULTS=$RESULTS SG_TABLES=$TABLES SG_MODELS=$MODELS_DIR SG_DATA=$DATA

# ---------------------------------------------------------------- helpers
log()  { printf '[run.sh] %s\n' "$*"; }
run()  { if [ $DRY = 1 ]; then printf '  $ %s\n' "$*"; else "$@"; fi; }
py()   { run "$VENV/bin/python" "$@"; }
have() { [ $FORCE = 0 ] && [ -e "$1" ] && { log "skip, exists: $1"; return 0; }; return 1; }
domain_of() { case $1 in synt-*) echo synt;; arrhythmia-*) echo arrhythmia;; audio-*) echo audio;; *) echo sleep;; esac; }
cfg()  { echo "configs/$(domain_of "$1")/$1_$2.yaml"; }              # cfg <model> sg|stft|sg_df5|sg_df2
result_file() {  # sleep names its SG results after the run parameters; the other domains use <explainer>_results.json
  local m=$1 e=$2 f
  if [ "$(domain_of "$m")" = sleep ]; then if [ "$e" = SG ]; then f="SG-shapley-fs0.5-s10-p20"; else f="$e"; fi; else f="${e}_results"; fi
  if [ "$(domain_of "$m")" = synt ]; then echo "$RESULTS/$m/$f.json"; else echo "$RESULTS/$m/$f${CHUNK_N:+_chunk${CHUNK_IDX}of${CHUNK_N}}.json"; fi; }
verify_models() { py - "$MODELS_DIR" <<'EOF'
import hashlib, pathlib, sys
d = pathlib.Path(sys.argv[1]); bad = []
if not (d / "SHA256SUMS").exists():
    print("no SHA256SUMS in", d, "- checkpoints not verified"); sys.exit(0)
for line in (d / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split()
    if hashlib.sha256((d / name).read_bytes()).hexdigest() != digest: bad.append(name)
sys.exit("checkpoint sha256 mismatch: %s" % bad if bad else 0)
print("checkpoints verified:", d)
EOF
}
need_physioex_data() { [ -n "${PHYSIOEX_DATA:-}" ] || { log "PHYSIOEX_DATA is not set (sleep domain skipped)"; return 1; }; }
has_mass() { [ -d "${PHYSIOEX_DATA:-/nonexistent}/MASS/Original/SS03" ]; }
confirm_full() {
  local h=0 d m
  for d in $DOMAINS; do for m in $(models_of "$d"); do h=$(python3 -c "print(round($h+$(full_h "$m"),1))"); done; done
  log "full benchmark on [$DOMAINS]: about $h GPU-hours on one A100 (SG on Speech and Sleep-EDF dominate)."
  log "Use --domains to subset and --chunk-idx/--chunk-n to spread a domain over several GPUs."
  [ $YES = 1 ] || [ $DRY = 1 ] && return 0
  read -r -p "proceed? [yes/N] " a; [ "$a" = yes ] || exit 1; }

# ---------------------------------------------------------------- stages
stage_setup() {
  python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "python >= 3.11 is required (physioex)")'
  have "$VENV/bin/python" || run python3 -m venv "$VENV"
  py -m pip install -q -U pip
  py -m pip install -q -r requirements.txt
  py - <<'EOF'
import physioex, torch, torchaudio, wfdb
from physioex.explain.posthoc.spectralgradients import SpectralGradients
from physioex.explain.posthoc.vistdft import STFTIntegratedGradients, STFTSaliency, STFTInputXGradient
from physioex.explain.posthoc.metrics import complexity, infidelity, tf_concentration, localization
from physioex.explain.posthoc.filters import lowpass_filter, highpass_filter
from physioex.models import load_from_pretrained
from physioex.data.datasets import get_dataset
from physioex.data.presets import get_preset
from torchaudio.datasets import SPEECHCOMMANDS
print("physioex", physioex.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
EOF
  verify_models; }

stage_data() {
  local d
  for d in $DOMAINS; do case $d in
    synt)       have "$DATA/synt/setup2/synt_class=3.npy" || py synt/data.py --n-samples 1000 --seed 42 --out "$DATA/synt";;
    arrhythmia) have "$DATA/mitdb/234.atr" || py arrhythmia/data.py --download "$DATA/mitdb";;
    audio)      have "$DATA/speech/SpeechCommands/speech_commands_v0.02/testing_list.txt" || py audio/train.py --download-only --data-root "$DATA/speech";;
    sleep)      need_physioex_data || continue
                if [ -z "$(ls "$PHYSIOEX_DATA/physionet-sleep-data" 2>/dev/null)" ]; then
                  command -v wget >/dev/null 2>&1 || { log "wget is needed to mirror Sleep-EDF (https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)"; exit 2; }
                  run mkdir -p "$PHYSIOEX_DATA/physionet-sleep-data"
                  run wget -r -N -c -np -nH --cut-dirs=4 -A '*.edf' -P "$PHYSIOEX_DATA/physionet-sleep-data" https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
                fi
                has_mass || log "MASS (SS3) requires a data-use agreement: place it under \$PHYSIOEX_DATA/MASS/Original/SS03/. The Sleep-MASS rows of the paper come from 'run.sh results'."
                py -c "from physioex.models import load_from_pretrained as L; L('tsinalis-2016'); L('chambon2018'); print('pretrained sleep models cached')";;
    *) log "unknown domain $d"; exit 2;;
  esac; done; }

stage_train() {   # optional; the shipped checkpoints in models/ are never overwritten
  local d out=models-retrained
  run mkdir -p $out
  for d in $DOMAINS; do case $d in
    synt)       py synt/train.py --data-root "$DATA/synt-train" --n-samples 2000 --seed 0 --epochs 10 --out $out;;
    arrhythmia) py arrhythmia/train.py --data-dir "$DATA/mitdb" --out $out/arrhythmia-cnn.pth;;
    audio)      py audio/train.py --data-root "$DATA/speech" --out $out/audio-wavcnn.pt;;
    sleep)      log "sleep models are pretrained (Hugging Face 4rooms/physioex); nothing to train";;
  esac; done
  log "to explain the retrained models: export SG_MODELS=$out"; }

stage_benchmark() {
  [ -n "$MODE" ] || { log "benchmark needs --quick or --full"; exit 2; }
  verify_models
  [ "$MODE" = full ] && confirm_full
  local chunk=() d m e n kind
  [ -n "$CHUNK_N" ] && chunk=(--chunk-idx "$CHUNK_IDX" --chunk-n "$CHUNK_N")
  for d in $DOMAINS; do
    if [ "$MODE" = full ]; then n=999999; else n=$(quick_n "$d"); fi
    [ $d = sleep ] && { need_physioex_data || continue; }
    if [ $d = synt ] && [ -n "$CHUNK_N" ] && [ "$CHUNK_IDX" != 0 ]; then log "synt is not chunked: run it from chunk 0 only"; continue; fi
    for m in $(models_of "$d"); do
      if [ $m = chambon2018 ] && ! has_mass; then log "skip $m (MASS not available)"; continue; fi
      for e in $EXPLAINERS; do
        if [ $e = SG ]; then kind=sg; else kind=stft; fi
        have "$(result_file $m $e)" && continue
        if [ $d = synt ]; then
          py $d/benchmark.py --config "$(cfg $m $kind)" --explainer $e --max-samples $n --gpu $GPU --seed 42 --output-dir "$RESULTS/$m"
        else
          py $d/benchmark.py --config "$(cfg $m $kind)" --explainer $e --max-samples $n --gpu $GPU --seed 42 --output-dir "$RESULTS/$m" ${chunk[@]+"${chunk[@]}"}
        fi
      done
      [ $d = synt ] && py synt/compute_localization.py --setups "${m#synt-setup}" --results "$RESULTS" --data-root "$DATA/synt"
    done
  done
  # band-width sensitivity (Table 6): synthetic setups 1-2 at 5 Hz, MASS at 2 Hz on chunk 0 of 8 (shares the fine run's sample cache)
  local s
  case " $DOMAINS " in *" synt "*)
    for s in 1 2; do
      if [ "$MODE" = full ]; then n=999999; else n=$(quick_n synt); fi
      have "$RESULTS/dftest/synt-setup$s/SG_results.json" || py synt/benchmark.py --config "configs/synt/synt-setup${s}_sg_df5.yaml" --explainer SG --max-samples $n --gpu $GPU --seed 42 --output-dir "$RESULTS/dftest/synt-setup$s"
    done;; esac
  case " $DOMAINS " in *" sleep "*)
    if has_mass; then
      if [ "$MODE" = full ]; then n=999999; else n=$(quick_n sleep); fi
      have "$RESULTS/chambon2018/SG-shapley-fs2.0-s10-p20_chunk0of8.json" || py sleep/benchmark.py --config configs/sleep/chambon2018_sg_df2.yaml --explainer SG --max-samples $n --gpu $GPU --seed 42 --output-dir "$RESULTS/chambon2018" --chunk-idx 0 --chunk-n 8
    fi;; esac; }

stage_merge() {
  local f; shopt -s nullglob
  for f in "$RESULTS"/*/*_chunk0of*.json; do py shared/merge_chunks.py "${f%_chunk0of*.json}"; done
  shopt -u nullglob; }

stage_results() {   # the paper's per-sample results (75 JSON files, 101 MB): tables without a GPU
  have "$RESULTS/.paper-results" && return 0
  run mkdir -p "$RESULTS"
  have "$RESULTS/paper-results.tar.gz" || run curl -fL -o "$RESULTS/paper-results.tar.gz" "$RESULTS_URL_GH" || run curl -fL -o "$RESULTS/paper-results.tar.gz" "$RESULTS_URL_ZENODO"
  [ $DRY = 1 ] || py -c "import hashlib,sys; h=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); sys.exit(0 if h==sys.argv[2] else 'sha256 mismatch: '+h)" "$RESULTS/paper-results.tar.gz" "$RESULTS_SHA256"
  run tar -xzf "$RESULTS/paper-results.tar.gz" --strip-components=1 -C "$RESULTS"     # archive root is results/
  run touch "$RESULTS/.paper-results"; }

stage_tables() {
  stage_merge
  run mkdir -p "$TABLES"
  py shared/generate_comparison_table.py            # Table 3 (comparison) + Table 4 (localisation) + comparison.csv
  py shared/effect_sizes.py                         # Table 5
  py shared/dftest_compare.py                       # Table 6
  if [ "$TABLES" = tables ] && [ $DRY = 0 ] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then git diff --stat -- tables/; fi; }

stage_figures() {
  run mkdir -p figures
  local tex=(); kpsewhich newtxtext.sty >/dev/null 2>&1 || tex=(--no-tex)     # Times fonts of the paper; else matplotlib text
  have figures/fig1_schema.pdf || py shared/fig1_schema.py ${tex[@]+"${tex[@]}"}
  if [ $NOGPU = 1 ] || { [ $FORCE = 0 ] && [ -e figures/k_convergence.json ]; }; then py synt/k_convergence.py --from-json
  else py synt/k_convergence.py --gpu $GPU; fi
  [ $NOGPU = 1 ] && { log "skip figures/sleep_epoch_bands.pdf (needs a GPU and Sleep-EDF)"; return 0; }
  need_physioex_data && { have figures/sleep_epoch_bands.pdf || py sleep/plot_sleep_epoch_bands.py --freq-step 0.5 --epoch-idx 44666 --chunk 4 --gpu $GPU; }
  return 0; }

stage_all() { stage_setup; stage_data; stage_benchmark; stage_tables; stage_figures; }

case $STAGE in
  setup|data|train|benchmark|merge|results|tables|figures|all) "stage_$STAGE";;
  *) log "unknown stage: $STAGE"; exit 2;;
esac
