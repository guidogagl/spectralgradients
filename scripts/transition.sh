#!/bin/bash

#SBATCH --account=lp_biomed_mdv
#SBATCH --clusters=genius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_p100_debug
#SBATCH --time=00:30:00 
#SBATCH --mem-per-cpu=32G
#SBATCH --output=output/logs/transition_job_%j.log  # Todo: set this to the ck


. /data/leuven/365/vsc36564/miniconda3/etc/profile.d/conda.sh
conda activate specgrad

cd /data/leuven/365/vsc36564/spectralgradients/

srun ../miniconda3/envs/specgrad/bin/python src/sleep/transition.py  2>&1
