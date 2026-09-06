#!/bin/bash

mainPy="/home/zwang182/MARL/test_repos/MAG/train.py"
seeds=(0 1 2 3)
# alg="cmbm"
# alg2="maser"
# seed=0
# t_max=2005000
# useCuda=True
# usetb=True
# save_model=True
# notsm=False
nworkers=2
envs_mag=(3s_vs_3z 3s_vs_4z 5m_vs_6m 8m_vs_9m corridor MMM MMM2)
envs_over24=(2c_vs_64zg so_many_banelings 27m_vs_30m)
e1="3s_vs_5z"
e2="so_many_baneling"
# envs_over24=(3s5z_vs_3s6z 2c_vs_64zg)
# envs_s_unfin=(3m 2m_vs_1z 2s_vs_1sc 3s_vs_3z)
# envs_under24=(3s_vs_4z 3s_vs_5z 5m_vs_6m 8m 3s5z 1c3s5z 8m_vs_9m 10m_vs_11m MMM MMM2 3s5z_vs_3s6z corridor)

#for e in "${envs_mag[@]}"
#do
for s in "${seeds[@]}"
  do
    sbatch run.sh ${mainPy} ${e1} ${nworkers}
done
#done
#for e in "${envs_over24[@]}"
#do
for s in "${seeds[@]}"
  do
     sbatch run2.sh ${mainPy} ${e2} ${nworkers}
done
#done
#sbatch run.sh ${mainPy} ${e1} ${nworkers}
# sbatch run.sh ${mainPy} ${alg} ${e2} ${seeds[1]} ${t_max} ${useCuda} ${usetb} ${notsm} ${ckpts[1]}
# sbatch run.sh ${mainPy} ${alg} ${e2} ${seeds[2]} ${t_max} ${useCuda} ${usetb} ${notsm} ${ckpts[2]}
# sbatch cc_runone_24h_marlp1.sh ${mainPy} ${alg} ${e2} ${seeds[3]} ${t_max} ${useCuda} ${usetb} ${notsm} ${ckpts[3]}
# sbatch run.sh ${mainPy} ${alg} ${e2} ${seeds[4]} ${t_max} ${useCuda} ${usetb} ${notsm} ${ckpts[4]}
# for s in "${seeds[@]}"
# do
#   sbatch run.sh ${mainPy} ${alg} ${e1} ${s} ${t_max} ${useCuda} ${usetb} ${save_model}
# done
# echo "Done"

