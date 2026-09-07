#!/bin/bash
#SBATCH --account=rrg-dpmeger
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/scratch/zwang182/three-baselines-valliappan/mag_profile/slurm-%j.out
# Profiles ONE learner.step() of MAG and of MAMBA (same code, use_MPCmodel flag) with the
# DEFAULT hyperparameters (60 model epochs / 60 m_r epochs / 4 actor epochs) on a full H100,
# plus a CPU run of the same on this node's 8 cores for comparison. No SC2, no ray workers.
module purge
module load StdEnv/2023
module load python/3.10 cuda/12.2 scipy-stack
source /home/zwang182/MARLenv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
D=/scratch/zwang182/three-baselines-valliappan/mag_profile
cd /scratch/zwang182/three-baselines-valliappan/MAG
export PYTHONPATH=$PWD:$D/stubs
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "########## MAMBA (use_MPC=0), defaults 60/60/4, GPU";  python $D/profile_learner.py 3s_vs_4z 0 60 60 4 8 cuda 2>&1 | grep -v Warning
echo "########## MAG   (use_MPC=1), defaults 60/60/4, GPU";  python $D/profile_learner.py 3s_vs_4z 1 60 60 4 8 cuda 2>&1 | grep -v Warning
echo "########## MAG   (use_MPC=1), reduced 10/10/1, CPU 8 cores (for CPU-vs-GPU ratio)"; python $D/profile_learner.py 3s_vs_4z 1 10 10 1 8 cpu 2>&1 | grep -v Warning
