#!/bin/bash
# Gate F2: profile ONE learner.step() per repo on a GPU (H100 3g.40gb = the decided Rorqual allocation).
# Prints per-phase time and per-phase peak GPU memory. Usage: sbatch cc_mag_profile_gpu.sh
#SBATCH --account=rrg-dpmeger
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/zwang182/three-baselines-valliappan/mag_profile/slurm-profile-%j.out
module purge
module load StdEnv/2023
module load python/3.10 cuda/12.2 scipy-stack
source /home/zwang182/MARLenv/bin/activate
export OMP_NUM_THREADS=2
D=/scratch/zwang182/three-baselines-valliappan/mag_profile
nvidia-smi --query-gpu=name,memory.total --format=csv
for R in MAG MAG_2; do
  cd /scratch/zwang182/three-baselines-valliappan/$R; export PYTHONPATH=$PWD:$D/stubs
  echo "########## $R, decided config (as in repo), GPU";   python $D/profile_learner.py 3s_vs_4z $([ $R = MAG ] && echo 1 || echo 0) 20 0 4 2 cuda 2>&1 | grep -v Warning
done
cd /scratch/zwang182/three-baselines-valliappan/MAG; export PYTHONPATH=$PWD:$D/stubs
echo "########## MAG, SHIPPED hyperparameters on the new code (for the before/after on GPU)"
MODEL_BATCH_SIZE=40 ROLLOUT_LEN=15 MPC_H=6 N_TRAJS=4 PPO_MINIBATCH=2000 python $D/profile_learner.py 3s_vs_4z 1 60 0 4 2 cuda 2>&1 | grep -v Warning
cd /scratch/zwang182/three-baselines-valliappan/MABL/mabl; export PYTHONPATH=$PWD:$D/stubs
echo "########## MABL, decided config, GPU"; python $D/profile_learner.py 3s_vs_4z 0 20 0 4 2 cuda 2>&1 | grep -v Warning
