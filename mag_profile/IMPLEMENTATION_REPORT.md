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
