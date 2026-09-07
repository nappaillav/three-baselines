#!/bin/bash
# MAG on Narval: A100 3g.20gb slice, 8 cores, 4 workers (DECISIONS.md). Usage: sbatch cc_mag_narval.sh <map> [repo_dir]   (env STEPS=<n> overrides the 2M budget)
# Verify the MIG gres name on Narval first:  sinfo -o "%G" | sort -u   (expected a100_3g.20gb)
#SBATCH --account=rrg-dpmeger
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=11:59:00
MAP=${1:-3s_vs_4z}
REPO=${2:-$HOME/projects/three-baselines/MAG}          # <- colleague's checkout; adjust
VENV=${VENV:-$HOME/env/mbmarlEnv}                      # <- colleague's venv (has wandb); adjust
module purge
module load StdEnv/2023
module load python/3.10 cuda/12.2
source $VENV/bin/activate
export SC2PATH=${SC2PATH:-$HOME/scratch/MARL/StarCraftII/StarCraftII}
export OMP_NUM_THREADS=1        # ray workers stay single-threaded; the driver sets its own torch threads (2)
export WANDB_MODE=offline       # compute nodes have no internet; `wandb sync` afterwards on a login node
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv
STEPS=${STEPS:-2000000}
python train.py --env=starcraft --env_name=$MAP --n_workers=4 --steps=$STEPS
