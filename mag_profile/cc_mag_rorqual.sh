#!/bin/bash
# MAG on Rorqual: H100 3g.40gb slice, 8 cores, 4 workers (DECISIONS.md). Usage: sbatch cc_mag_rorqual.sh <map> [repo_dir]
#SBATCH --account=rrg-dpmeger
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/zwang182/three-baselines-valliappan/mag_profile/slurm-mag-%j.out
MAP=${1:-3s_vs_4z}
REPO=${2:-/scratch/zwang182/three-baselines-valliappan/MAG}
module purge
module load StdEnv/2023
module load python/3.10 cuda/12.2 scipy-stack
source /home/zwang182/MARLenv/bin/activate
export PYTHONPATH=$REPO:/scratch/zwang182/three-baselines-valliappan/mag_profile/stubs   # stubs: no-op wandb (MARLenv has none); drop if the venv has wandb
export SC2PATH=/home/zwang182/MARL/marl_project1/3rdparty/StarCraftII
export OMP_NUM_THREADS=1        # ray workers stay single-threaded; the driver sets its own torch threads (2)
export WANDB_MODE=offline
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py --env=starcraft --env_name=$MAP --n_workers=4
