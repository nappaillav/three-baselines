# Why MAG runs slowly, and what to do about it

Scope: the MAG baseline in `three-baselines-valliappan/MAG`, map 3s_vs_4z, target 1M env steps.
Evidence: code inspection, a CPU profile on Rorqual (2026-09-06), and the colleague's two A100 wandb runs
(Sep 4: 37 s per episode, Sep 6: 52 s per episode; ~40-80 env steps per episode).

Nothing in the fork is misconfigured. Hyperparameters equal the authors' defaults, the GPU is in use,
and training converges. The slowness has six causes, ordered by impact.

## 1. The 1M-step target is 10x the published budget
The MAG paper's 3s_vs_4z curves stop at 100k steps (25k-500k across maps). At 37-52 s per episode,
1M steps is 3-8 days per run; 100k steps is 8-20 hours.
**Solution:** evaluate at the paper's budget per map (100k for 3s_vs_4z). Reserve 1M-step runs for
maps the paper ran that far, if any. This alone makes the runs practical without touching the code.

## 2. A full learner update after every episode (MAMBA/MAG design)
`N_SAMPLES=1`: every episode (40-200 env steps) triggers 60 world-model epochs, 60 model-error
predictor epochs, 4 imagination rollouts and 120 PPO updates. On the A100 that is 11 s + 9 s + 16-31 s.
The environment itself costs <1 s per episode, so >97% of wall time is the learner.
**Solution:** if deviating from the paper is acceptable, raise `N_SAMPLES` (update every k episodes)
or lower `MODEL_EPOCHS` / `m_r_predictor_epochs`; the time scales linearly with those counts. If a
faithful baseline is required, keep the defaults and accept cause 2 as fixed; apply causes 3-6 instead.

## 3. MPC planning inside imagination (MAG-specific)
Each of the 15 imagination steps runs a 6-step MPC lookahead over 4 candidate trajectories, i.e. 90
sequential world-model passes per rollout at 4x batch, 4 rollouts per episode. That is ~25x MAMBA's
imagination work and 45-60% of the cycle (16-31 s of 37-52 s on the A100; 268 s of 419 s on CPU).
**Solution:** cost is proportional to `rollout_max_length x MPCHorizon x n_trajs`. The shipped
values (15, 6, 4) are already cheaper than the paper's Table 1 (H=10, L=4 for this map), so there is
no free reduction; only reduce them if the study is not about reproducing MAG exactly. Where MAG is
not required, run MAMBA (`use_MPCmodel=False`): ~3.7x cheaper per episode.

## 4. Redundant computation in the model-error-predictor phase (implementation)
`DreamerLearner.step` recomputes the world-model rollout twice per predictor epoch
(`get_model_loss_for_m_r_training`, then `m_r_perdictor_loss` again on the same batch), and discards
the per-step model loss that `train_model` already computes (`loss, _ = model_loss(...)`; the unused
`DreamerMemory.sample_all` shows the original intent to reuse it).
**Solution:** cache `model_loss_per_step` and the posterior features from the 60 model-training
epochs and train the predictor on them; remove the second rollout. Same algorithm, removes most of
the 9 s phase: ~25% of the A100 cycle, ~14% on CPU.

## 5. Host-bound execution: tiny kernels and per-step syncs
The driver process averaged <1 CPU core and the GPU ~47% power. The RSSM loops are Python-driven
(20 sequential steps per model epoch, 90 per MPC rollout) and `MPCPredict` syncs the GPU every
planning step (`.cpu().numpy()`, boolean-mask assignment `pi[avail==0]=...`). CPU->A100 speedup is
only ~11x, and a larger GPU will not help.
**Solutions (no change to the math):** keep `traj_losses` on the GPU and take the argmin once per
step; replace masked assignment with `torch.where`; run the 4 actor epochs' rollouts as one batch of
4x40 sequences (per-step cost is latency-bound, so 4x fewer sequential steps at ~the same per-step
time); optionally CUDA graphs / `torch.compile` on the transition step.

## 6. Allocation-dependent overhead (node, cores, threads)
With identical config, the Sep 6 run's MPC phase was 1.9x slower than the Sep 4 run's (7.7 s vs
4.1 s per actor iteration, from the first cycle) while other phases matched, and the GPU's total
memory reading was 3x higher throughout. That points at host-side effects: `torch.set_num_threads(10)`
is hardcoded in `train.py` regardless of the cores granted; cores may not be local to the GPU; or
another process shared the GPU.
**Solutions:** set threads from `SLURM_CPUS_PER_TASK`; request at least that many cores per GPU;
check `nvidia-smi` at job start for foreign processes; use GPU-local core binding where the cluster
supports it. Two workers are enough — the runner is synchronous and workers idle during learning, so
extra workers only cost cores.

## Expected effect (per episode, A100, relative to 37 s)
| action | cost | saving |
|---|---|---|
| Paper budget instead of 1M (cause 1) | none | 10x fewer episodes |
| Reuse model losses for the predictor (4) | small code change | ~9 s (~25%) |
| Remove syncs, batch the 4 rollouts (5) | moderate code change | several s, node-dependent |
| Fix threads / cores / GPU locality (6) | job script | up to 15 s on a bad allocation |
| Lower epochs or N_SAMPLES (2) | deviates from paper | linear |
| Lower MPC H/L/rollout length (3) | deviates from paper | up to ~45% |
