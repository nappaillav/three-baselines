Currently, I am running experiments to verify whether the GPU is being utilized and to measure the time required to complete 1M steps. I am running these experiments from the MAG folder.

```
module --force purge && module load StdEnv/2023 python/3.10 cuda/12.2 rust/1.85.0
source ~/env/mbmarlEnv/bin/activate
export SC2PATH=~/scratch/MARL/StarCraftII/StarCraftII
cd ~/projects/three-baselines/MAG
wandb offline

python train.py --env=starcraft --env_name=3s_vs_4z --n_workers=2
```