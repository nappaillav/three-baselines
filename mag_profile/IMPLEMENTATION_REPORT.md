# Implementation report — MAG speed-up (branch `speedup`)

Written incrementally, one section per verified step. Environment for all local checks: MARLenv
(unmodified) + `mag_profile/stubs` on PYTHONPATH (no-op wandb), CPU only, login node, 8 threads.
Profiler: `mag_profile/profile_learner.py` (synthetic buffer, map 3s_vs_4z dims: 3 agents, obs 42, 10 actions).

## MAG

### Step 0 — profiler instrumented, baseline recorded
`profile_learner.py` now prints a per-phase peak-GPU-memory column on cuda (`reset_peak_memory_stats`
/ `max_memory_allocated` around each wrapped call) and accepts env-var overrides
(`MODEL_BATCH_SIZE, BATCH_SIZE, ROLLOUT_LEN, MPC_H, N_TRAJS, PPO_MINIBATCH, PPO_EPOCHS, MR_UPDATES, N_SAMPLES`).
Verified it still runs on CPU.

Baseline, shipped code, shipped config at reduced epochs (MODEL 10 / m_r 10 / EPOCHS 1), CPU 8 threads:
```
TOTAL learner.step = 90.1s
1 model update (fwd+bwd, per epoch)        10   9.48   0.948/call
2a m_r: no-grad model fwd                  10   5.02   0.502
2b m_r: predictor update                   10   4.52   0.452
3 actor_rollout (total)                     1  63.07
3b   imagination rollout_policy             1  59.51
3b-i   MPCPredict calls                    15  59.28   3.952
4a-c PPO (30 minibatches)                       7.4
```
Baseline, shipped code, decided overrides (MODEL 20 x batch 120, imag 5 / H 3 / L 2, m_r 10, EPOCHS 4):
```
TOTAL learner.step = 79.8s
1 model update                             20  31.57   1.578/call   (batch 120 is compute-bound on CPU)
2a+2b m_r                                  10  17.58
3 actor_rollout (total)                     4  22.04   5.509/call
3b-i   MPCPredict calls                    20  16.67   0.833
4 PPO (40 minibatches)                          8.0
```

### Step A — predictor trained on model-phase losses (exact)
Changes: `agent/optim/loss.py::model_loss` now also returns the per-step model error **including the
`dis` term on steps 1:** (the label the old `get_model_loss_for_m_r_training` produced) and the prior
features `prior.get_features()` (the input the old `m_r_perdictor_loss` recomputed); new tensor-only
`m_r_predictor_loss(m_r_predictor, mr_input, mr_label)`; old `get_model_loss_for_m_r_training` and
`m_r_perdictor_loss` deleted. `DreamerLearner.train_model` returns `(losses, mr_input, mr_label)`;
`train_m_r_predictor(mr_input, mr_label)`; `step` runs `m_r_updates_per_model_epoch` (new config, =1)
predictor updates right after each model epoch and the separate 60-epoch loop is gone.

Exactness (fixed seed, untrained model, same batch, old vs new in one process):
```
label max|diff| 1.9e-06  allclose: True   (shape (18, 40, 3, 1))
input max|diff| 0.0      allclose: True   (shape (18, 40, 3, 1280))
STEP A EXACTNESS: PASS
```
Profiler after A (shipped config, 10 model epochs, EPOCHS 1): m_r phase 9.5 s -> 0.17 s (10 predictor
updates at 17 ms); everything else unchanged. TOTAL 90.1 s -> 75.2 s.

### Step B — MPC loop without host syncs (exact)
Changes in `networks/dreamer/rnns.py`: `RSSMTransition.para_predict` (list-of-states API) replaced by
`para_predict_flat(prev_actions, stoch, deter)` on flat `(n_trajs*B, n_agents, .)` tensors; `MPCPredict`
rewritten — candidate trajectories expanded once, `traj_losses` kept on the device, `argmin` once per
imagination step, first-step prediction gathered by index (no `.cpu().numpy()`, no per-step Python
list/stack rebuild), returns only the state; `rollout_policy` drops the `minlosses`/`ranlosses`
bookkeeping (only fed a commented print) and uses `masked_fill`. `masked_fill` also in
`agent/optim/loss.py::actor_loss` and `agent/controllers/DreamerController.py::step`.

Exactness (same model/actor/predictor, same batch, same torch seed, old module loaded from a saved copy):
```
steps=5  H=3 K=2: stoch/deter/logits/actions/av_actions/old_policy all torch.equal -> True
steps=15 H=6 K=4: all torch.equal -> True
STEP B EXACTNESS: PASS
```
Profiler after B on CPU: MPCPredict 3.65 -> 3.92 s/call, i.e. **no change on CPU** (within noise). This
is expected: the removed costs are host<->device syncs and launch gaps, which only exist on a GPU. The
GPU effect is measured by gate F2 (job script below).

### Step C — one batched imagination rollout per cycle
Changes: `DreamerLearner.step` samples `EPOCHS * BATCH_SIZE` (=160) sequences once and calls
`train_agent` once (was `EPOCHS` sequential rollouts of 40); PPO minibatch size is now
`config.PPO_MINIBATCH` (new, 1000; was the literal 2000); `actor_loss`'s unused `obs_as_pol_in` reshape
no longer hardcodes `BATCH_SIZE`. `EPOCHS` keeps its name as the batch multiplier.

Profiler after C (shipped config, MODEL 10, EPOCHS 4): `actor_rollout` called **once** on 160 sequences
(imagined states (15, 2880, 3, 1024)); 205 PPO minibatches of <=1000 rows (40,320 rows / 1000 x 5 PPO
epochs). On CPU the single wide rollout costs 246 s = ~4 x the 62 s single-batch rollout, i.e. no CPU
saving — CPU is compute-bound. The saving (4x fewer sequential latency-bound passes) is a GPU effect;
gate F2 measures it.

### Step D — decided hyperparameters (`configs/dreamer/DreamerLearnerConfig.py`)
`MODEL_BATCH_SIZE 40->120`, `MODEL_EPOCHS 60->20`, `N_SAMPLES 1->500` (transitions), imagination
`rollout_min/max_length 15->5`, `MPCHorizon 6->3`, `n_trajs 4->2`; plus the step A/C knobs
`m_r_updates_per_model_epoch=1`, `PPO_MINIBATCH=1000`. `MODEL_LR` unchanged (5e-4).

Profiler after D, config exactly as in the repo (CPU, 8 threads) — one cycle now serves 500 env steps:
```
TOTAL learner.step = 61.1s          (baseline shipped: ~419 s per cycle, one cycle per ~70-step episode)
1 model update                 20   29.73   1.486/call   (batch 120; CPU compute-bound)
2b m_r: predictor update       20    0.91
3 actor_rollout (total)         1   21.61   (single 160-sequence rollout, imagination (5, 2880, 3, 1024))
3b-i   MPCPredict calls         5   16.89   3.38/call
4 PPO: 60 minibatches (11,520 rows / 1000 x 5 epochs)   8.2
```
Per env step on CPU: 419/70 = 6.0 s -> 61/500 = 0.12 s (~50x). GPU figures: gate F2.

### Step E — driver / ray / job scripts
`train.py`: `torch.set_num_threads(2)`; `SC2PATH` must come from the environment (raises a clear
`EnvironmentError` otherwise; the hardcoded per-user path is gone). `agent/runners/DreamerRunner.py`:
`ray.init(num_cpus=n_workers, object_store_memory=512 MB, include_dashboard=False)`.
Job scripts in `mag_profile/`: `cc_mag_rorqual.sh` (H100 3g.40gb, 8 cores, 4 workers, 14 h),
`cc_mag_narval.sh` (A100 3g.20gb, 8 cores, 4 workers; paths/venv for the colleague's checkout, MIG gres
name to be confirmed with `sinfo`), `cc_mag_profile_gpu.sh` (gate F2: per-phase time + peak memory for
MAG, MAG_2, MABL and the shipped-vs-decided MAG comparison, 30 min).
`sbatch --test-only` (validates, submits nothing): both Rorqual scripts accepted.

### End-to-end smoke run (gate F1 for the whole chain)
`train.py --env=starcraft --env_name=3m --n_workers=1`, CPU, 16 episodes, no repo edits (a wrapper
monkeypatched `Experiment.episodes`, `DEVICE='cpu'` and the thread count):
```
Started a local Ray instance.
[smoke] train_agent done in 35.4s; model losses this cycle: first=13.340 last=7.953 all_finite=True
[smoke] finished 16 episodes in 85s      exit 0
```
One learner cycle fired once 500 transitions had accumulated (N_SAMPLES), ran 20 model epochs, the
predictor updates, the batched imagination rollout and PPO, and the workers resumed. No NaN/Inf.

Not run locally (needs a GPU job, user submits): gate F2. Not run (needs hours): F3-F5.

## MAG_2 (MAMBA)

MAG_2's source was byte-identical to MAG's except `use_MPCmodel=False` and the colleague's wandb
toggles (MAG_2 never enabled wandb). Steps A, B, C, E were replayed from the MAG commits with
`git diff <step> -- MAG | git apply --directory=MAG_2` (A and E needed `-C1` because of the wandb-import
lines); step D's hunk did not apply (a blank line + the `use_MPCmodel=False` context) and was applied with
the same exact-match patch as in MAG. Commit order in MAG_2 is therefore A, B, C, E, D.
After replay, `diff -r MAG MAG_2` shows only: the wandb import/flag lines, `use_MPCmodel`, the two
MPC-knob comments, and the shell scripts — i.e. the shared code is identical, so the step A/B exactness
checks (which exercise code paths MAG_2 never runs) were not repeated.

Applicability: A (predictor) and B's MPC rewrite are dormant in MAG_2 (`use_MPCmodel=False`), but keeping
the files identical avoids divergence; B's `masked_fill` and C, D (non-MPC values), E apply fully.

Profiler, decided config as in the repo (`use_MPC=0`, CPU 8 threads):
```
TOTAL learner.step = 47.4s   (one cycle per 500 env steps)
1 model update                 20   30.45   1.522/call
3 actor_rollout (total)         1    7.57   (single 160-sequence rollout, imagination length 5)
4 PPO: 60 minibatches               8.7
```
Smoke run (`3m`, 1 worker, CPU): with 16 episodes no cycle fired — the 16 short random-policy episodes
summed to <500 transitions (N_SAMPLES); with 28 episodes:
```
[smoke] train_agent done in 21.1s; model losses this cycle: first=13.226 last=8.123 all_finite=True
map: 3m, cur_step: 616, incre_win_rate: 0.0
[smoke] finished 28 episodes in 70s      exit 0
```
