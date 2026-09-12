#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --job-name=27_mag_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=11:59:00
#SBATCH --array=0-1
#SBATCH --output="/home/chidamv/scratch/logs/log4/out/%x-%A_%a.out"
#SBATCH --error="/home/chidamv/scratch/logs/log4/log/%x-%A_%a.err"

start=$(date +%s)

### -------------------------
### Load modules
### -------------------------
module --force purge
module load StdEnv/2023 python/3.10 cuda/12.2 rust/1.85.0

### -------------------------
### Activate virtual environment
### -------------------------
source /home/chidamv/env/mbmarlEnv/bin/activate

### -------------------------
### Repository / Algorithm
### -------------------------

REPOS=(
    # "/home/chidamv/projects/def-dpmeger/chidamv/three-baselines/MABL/mabl"
    "/home/chidamv/projects/def-dpmeger/chidamv/three-baselines/MAG"
    "/home/chidamv/projects/def-dpmeger/chidamv/three-baselines/MAG_2"
)

ALGORITHMS=(
    # "MABL"
    "MAG"
    "MAG_2"
)

REPO=${REPOS[$SLURM_ARRAY_TASK_ID]}
ALGO=${ALGORITHMS[$SLURM_ARRAY_TASK_ID]}

cd "$REPO" || exit 1

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
### Experiment configuration
### -------------------------

MAP="27m_vs_30m"

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
echo "ALGORITHM:        $ALGO"
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
eval "$CMD"

### -------------------------
### Runtime
### -------------------------
echo "Experiment task $SLURM_ARRAY_TASK_ID ($ALGO, $MAP) finished."

end=$(date +%s)
runtime=$((end - start))

echo "Runtime: $((runtime / 3600)) hours, $(((runtime % 3600) / 60)) minutes, $((runtime % 60)) seconds"