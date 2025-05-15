srun    --account=lp_biomed_mdv \
        --clusters=genius \
        --partition=gpu_p100_debug \
        --ntasks-per-node=1 \
        --gpus-per-node=1 \
        --mem-per-cpu=32G \
        --time=00:30:00 \
        --pty bash -l