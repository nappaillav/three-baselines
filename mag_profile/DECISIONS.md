# Decisions for adapting the MAG baseline (recorded 2026-09-06)

Context: MAG as shipped costs 37-52 s of learner time per ~70-step episode on an A100 (see
`MAG_slow_runtime_summary.md`); the adaptation plan is `MAG_speedup_plan.md`. We are adapting the
baselines, not reproducing them, so changing their code is acceptable.

## Training budget and update schedule
- Budget: **2M environment steps** per run.
- Learner update every **500 transitions** (`N_SAMPLES=500`, per-500-step cycle), not every episode.

## Code changes (plan steps A-D)
- A: train the m_r predictor on the losses/features computed during model training (no extra rollouts).
- B: remove host syncs and list churn from the MPC loop (`MPCPredict`, `rollout_policy`).
- C: run the actor epochs as one batched imagination rollout; keep PPO gradient-step parity.
- D: the hyperparameters below.

## Hyperparameters
| parameter | shipped | decided |
|---|---|---|
| `MODEL_BATCH_SIZE` | 40 | **120** |
| `MODEL_EPOCHS` | 60 | **20** (same 2,400 sequences per cycle as 60 x 40) |
| `rollout_max_length` (imagination steps) | 15 | **5** |
| `MPCHorizon` | 6 | **3** |
| `n_trajs` | 4 | **2** |
| `N_SAMPLES` | 1 | **500** |
| torch threads in the driver (`train.py`) | hardcoded 10 | **2** (set explicitly; do not derive from core count) |

## Cluster allocation
- **Narval:** `A100 3g.20gb` slice, `--cpus-per-task=8`, `--n_workers=4`. Charged as 8/12 of an A100.
  Open item: confirm the actor-phase GPU peak fits in 20 GB before committing many runs.
- **Other clusters (Rorqual etc.):** `H100 3g.40gb` slice (`--gpus=h100_3g.40gb:1`),
  `--cpus-per-task=8`, `--n_workers=4`. Charged as 8/16 of an H100.
- Job-script details that go with this: `--mem` ~24-32G, `OMP_NUM_THREADS=1` for the ray workers,
  explicit `ray.init(num_cpus=4, object_store_memory=512MB)`, threads=2 in the driver.

## Expected cost (from the corrected estimate table; +/-30%)
| option | charge | cycle | 2M steps | charge x h |
|---|---|---|---|---|
| A100 3g.20gb, 8 cores, 4 workers | 0.67 | ~8.2 s | ~9.1 h | 6.1 |
| H100 3g.40gb, 8 cores, 4 workers | 0.50 | ~8.8 s | ~9.7 h | 4.9 |
(full GPU with 4 workers would be ~7.2 h at charge 1.0.)

## Verification gates before scaling out
1. Profiler (`profile_learner.py`, `cc_mag_profile_gpu.sh`) before/after each of A-D: phase times and
   per-phase peak GPU memory.
2. One 100k-step run on 3s_vs_4z per configuration: win rate within noise of the paper's MAG curve.
3. Check the wandb system stats of the first 4-worker job: driver ~1 core, no core oversubscription,
   cycle time matching the estimate.
