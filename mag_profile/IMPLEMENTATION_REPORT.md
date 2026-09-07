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
