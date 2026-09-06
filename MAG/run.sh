#!/bin/bash
#SBATCH --account=rrg-dpmeger
#SBATCH --gpus-per-node=1
#SBATCH --mem=30G
#SBATCH --oversubscribe
#SBATCH --time=71:00:00

module load StdEnv/2020
module load python/3.9 cuda/11.7 scipy-stack
source /home/zwang182/MBVDEnv/bin/activate

mainPy=$1
sc_env=$2
nworkers=$3
# alg=$4
# seed=$4
# t_max=$5
# useCuda=$6
# usetb=$7
# saveModel=$8
# ckpt=$9

GLIBC_PATH1="/home/zwang182/MBVDEnv/bin/"
GLIBC_PATH2="/home/zwang182/MARL/marl_project1/3rdparty/StarCraftII/Libs"
GLIBC_PATH3="/home/zwang182/MARL/marl_project1/3rdparty/StarCraftII/Versions/Base75689"
GLIBC_PATH4="/home/zwang182/MARL/marl_project1/3rdparty/StarCraftII/Versions/Base75689/SC2_x64"

setrpaths.sh --path=${GLIBC_PATH1}
setrpaths.sh --path=${GLIBC_PATH2}
setrpaths.sh --path=${GLIBC_PATH3}
setrpaths.sh --path=${GLIBC_PATH4}

python3 ${mainPy} --n_workers=${nworkers} --env="starcraft" --env_name=${sc_env} 

# echo "main.py=${mainPy} algorithm=${alg} smac_env=${sc_env} seed=${seed} t_max=${t_max} use_cuda=${useCuda} use_tensorboard=${usetb} save_model=${saveModel}"
# python ${mainPy} --config=${alg} --env-config=sc2 with env_args.map_name=${sc_env} env_args.seed=${seed} t_max=${t_max} \
#   use_cuda=${useCuda} use_tensorboard=${usetb} save_model=${saveModel}

