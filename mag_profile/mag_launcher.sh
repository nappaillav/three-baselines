#!/bin/bash

#SBATCH --account=def-dpmeger
#SBATCH --job-name=mag_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=11:59:00
#SBATCH --array=0-8
#SBATCH --output="/home/chidamv/scratch/logs/out/%x-%A_%a.out"
#SBATCH --error="/home/chidamv/scratch/logs/log/%x-%A_%a.err"

start=$(date +%s)

### -------------------------
### Load modules
### -------------------------
module --force purge
module load StdEnv/2023 python/3.10 cuda/12.2 rust/1.85.0

### -------------------------
### Activate virtual environment
### -------------------------
# VENV=${VENV:-/home/chidamv/env/mbmarlEnv}
source /home/chidamv/env/mbmarlEnv/bin/activate

### -------------------------
### Repository
### -------------------------
REPO=${1:-/home/chidamv/projects/def-dpmeger/chidamv/three-baselines/MAG}
cd $REPO

### -------------------------
### StarCraft II
### -------------------------
export SC2PATH=${SC2PATH:-$HOME/scratch/MARL/StarCraftII/StarCraftII}

### -------------------------
### Threading / WandB
### -------------------------
export OMP_NUM_THREADS=1
export WANDB_MODE=offline

wandb offline

### -------------------------
### Experiment configurations
### -------------------------
MAPS=(
    "27m_vs_30m"
    "3s5z_vs_3s6z"
    "MMM2"
    "3s_vs_4z"
    "3s_vs_5z"
    "6h_vs_8z"
    "corridor"
    "so_many_baneling"
    "2c_vs_64zg"
)

MAP=${MAPS[$SLURM_ARRAY_TASK_ID]}

### -------------------------
### Training configuration
### -------------------------
STEPS=${STEPS:-2000000}
N_WORKERS=4

CMD="python train.py \
    --env=starcraft \
    --env_name=$MAP \
    --n_workers=$N_WORKERS \
    --steps=$STEPS"

### -------------------------
### Print experiment information
### -------------------------
echo "================================================"
echo "JOB ID:           $SLURM_JOB_ID"
echo "ARRAY JOB ID:     $SLURM_ARRAY_JOB_ID"
echo "TASK ID:          $SLURM_ARRAY_TASK_ID"
echo "ENV:              $MAP"
echo "STEPS:            $STEPS"
echo "WORKERS:          $N_WORKERS"
echo "REPO:             $REPO"
echo "SC2PATH:          $SC2PATH"
echo "COMMAND:"
echo "$CMD"
echo "================================================"

### -------------------------
### GPU information
### -------------------------
nvidia-smi --query-gpu=name,memory.total --format=csv

### -------------------------
### Run experiment
### -------------------------
eval $CMD

### -------------------------
### Runtime
### -------------------------
echo "Experiment task $SLURM_ARRAY_TASK_ID ($MAP) finished."

end=$(date +%s)
runtime=$((end - start))

echo "Runtime: $((runtime / 3600)) hours, $(((runtime % 3600) / 60)) minutes, $((runtime % 60)) seconds"