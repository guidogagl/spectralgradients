#!/bin/bash

#SBATCH --account=lp_biomed_mdv
#SBATCH --clusters=genius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_p100
#SBATCH --time=01:30:00 
#SBATCH --mem-per-cpu=32G
#SBATCH --output=output/logs/compute_gradients_job_%j.log  # Todo: set this to the ck


. /data/leuven/365/vsc36564/miniconda3/etc/profile.d/conda.sh
conda activate specgrad

cd /data/leuven/365/vsc36564/spectralgradients/

srun ../miniconda3/envs/specgrad/bin/python src/sleep/compute_gradients.py  2>&1
