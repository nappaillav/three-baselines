# three-baselines — MAG, MAMBA, MABL adapted for fast SMAC runs

Branch `speedup`. Three model-based MARL baselines on StarCraft II (SMAC):

| folder | algorithm | entry point |
|---|---|---|
| `MAG/` | MAG (Wu et al., AAAI 2023) | `MAG/train.py` |
| `MAG_2/` | MAMBA (Egorov & Shpilman, 2022) — same code as MAG with `use_MPCmodel=False` | `MAG_2/train.py` |
| `MABL/mabl/` | MABL (bi-level latent world model) — single-process runner | `MABL/mabl/train.py` |

The shipped code runs a full learner update after **every** episode (60 world-model epochs, 60
model-error-predictor epochs, 4 MPC imagination rollouts, 120 PPO updates for ~70 env steps), which
measured 37–52 s per episode on an A100 and would need 6–12 days for 2M steps. This branch adapts the
baselines to run 2M steps in roughly 9–10 h on a 3g GPU slice. Analysis, decisions and the
implementation record are in `mag_profile/` (`MAG_slow_runtime_summary.md`, `DECISIONS.md`,
`IMPLEMENTATION_REPORT.md`).

## 1. What was changed

All changes preserve the algorithms' math unless marked *(training change)*.

**Learner schedule and hyperparameters — all three repos** *(training change)*, in
`configs/dreamer/DreamerLearnerConfig.py`:

| parameter | shipped | now | meaning |
|---|---|---|---|
| `N_SAMPLES` | 1 | **500** | learner update every 500 transitions instead of every episode |
| `MODEL_BATCH_SIZE` × `MODEL_EPOCHS` | 40 × 60 | **120 × 20** | same 2,400 sequences per update, 3× fewer (larger) gradient steps |
| imagination length (`rollout_max_length`; `HORIZON` in MABL) | 15 | **5** | shorter Dreamer rollouts |
| `MPCHorizon` / `n_trajs` (MAG only) | 6 / 4 | **3 / 2** | cheaper MPC planning |
| `PPO_MINIBATCH` (new) | literal 2000 | **1000** | PPO minibatch rows |
| `--steps` (new CLI flag in `train.py`) | 1e6 (MABL: unbounded) | **2,000,000** | environment-step budget |

**MAG-specific code (exact, verified equal to the original):**
- *Predictor on model-phase losses.* The model-error predictor is trained on the per-step losses and
  prior features already computed during world-model training, interleaved one update per model
  epoch. The separate 60-epoch predictor loop and its two extra world-model rollouts are removed
  (`agent/learners/DreamerLearner.py`, `agent/optim/loss.py`).
- *MPC loop without host syncs.* `MPCPredict` works on flat `(n_trajs·B)` batches with a device-side
  `argmin`; no `.cpu().numpy()` per planning step, no per-step Python list rebuilds; action masking uses
  `masked_fill` (`networks/dreamer/rnns.py`, `agent/optim/loss.py`, `agent/controllers/DreamerController.py`).
- *One batched imagination rollout per update* *(training change)*: the 4 actor epochs run as a
  single rollout over 4 × 40 sequences instead of 4 sequential ones.

**MAMBA (`MAG_2`)**: the same code changes as MAG (its shared files are byte-identical to MAG's apart
from `use_MPCmodel=False`); the MPC/predictor paths are inactive there.

**MABL**: no predictor and no MPC, so only `masked_fill`, the batched imagination rollout, the
hyperparameters above and the driver changes apply.

**Driver and job plumbing — all three repos** (`train.py`, `agent/runners/DreamerRunner.py`):
- torch threads fixed at 2 in the driver (the learner is latency-bound; more threads only steal cores
  from the workers); `OMP_NUM_THREADS=1` is set for the ray workers by the job scripts.
- `SC2PATH` must be exported; the hardcoded per-user paths are gone and `train.py` fails loudly without it.
- `ray.init` is bounded (`num_cpus=n_workers`, 512 MB object store) so it cannot exceed the job's memory.
- **wandb hardened (all three `train.py`)**: `wandb.init` is passed a settings object that disables
  wandb's *system-metrics monitor* (GPU/CPU/RAM sampler), and `wandb.log` is wrapped so a logging
  failure prints one warning and training continues instead of aborting. Reason: on Narval's MIG
  slices the GPU sampler of wandb 0.16's service process crashed after ~3.4 h
  (`Fatal Python error: none_dealloc`), the trainer then died on the next `wandb.log` with
  `BrokenPipeError`, and every MAG run stopped at 0.66–1.3M steps. Training metrics (`incre_win_rate`,
  `total_step`, `aver_step_reward`, losses) are logged exactly as before; only wandb's "System" panel
  is gone. The setting name is resolved at runtime, so this works with wandb 0.16 and current releases.
- `mag_profile/stubs/` holds no-op `wandb` and `setproctitle` modules for venvs that lack them (e.g.
  MARLenv); put the directory on `PYTHONPATH` as the Rorqual job script does.

**Verification done on CPU** (`mag_profile/tests/`, `mag_profile/IMPLEMENTATION_REPORT.md`): the
predictor labels/inputs match the old code to 2e-6 / exactly; the rewritten MPC loop reproduces the
old imagined states bit-for-bit; an end-to-end short run per repo completes one learner update with
finite losses. The GPU-side timing (gate F2) is measured by the profiling job below.

## 2. Running the baselines

### Resources (decided in `mag_profile/DECISIONS.md`)

| cluster | GPU | `--cpus-per-task` | workers | `--time` |
|---|---|---|---|---|
| Narval | `a100_3g.20gb:1` (A100 3g.20gb slice) | 8 | 4 | 11:59:00 |
| Rorqual and other H100 clusters | `h100_3g.40gb:1` (H100 3g.40gb slice) | 8 | 4 | 11:59:00 |

`--mem=24G`. Charged as 8/12 of an A100 or 8/16 of an H100. Expected wall time for 2M steps is
~9–10 h (±30%); there is **no checkpoint/resume** in these repos, so a run must finish within
11:59:00. If a first run comes in too close to the limit, lower the budget with `STEPS=` (see below)
rather than raising `--time`.

### Prerequisites
- A Python 3.10 venv with the packages in `requirements.txt` (see `setup_information.md`), plus `wandb`
  if you want logging. On a venv **without** wandb, keep `mag_profile/stubs` on `PYTHONPATH` as the
  Rorqual script does (no-op stub).
- StarCraft II 4.10 + SMAC maps, and `SC2PATH` exported to the `StarCraftII` directory.
- Compute nodes have no internet: run wandb offline (`WANDB_MODE=offline`, set by the scripts) and
  `wandb sync <repo>/wandb/offline-run-*` afterwards from a login node.

### Submit (Rorqual)
From `mag_profile/`, one job per algorithm/map/seed. The script takes the map and, optionally, the repo:

```bash
cd mag_profile
sbatch cc_mag_rorqual.sh 3s_vs_4z                                  # MAG
sbatch cc_mag_rorqual.sh 3s_vs_4z /scratch/zwang182/three-baselines-valliappan/MAG_2      # MAMBA
sbatch cc_mag_rorqual.sh 3s_vs_4z /scratch/zwang182/three-baselines-valliappan/MABL/mabl  # MABL
```

Set the account, venv path, `SC2PATH` and output path inside the script for your own checkout.
Optional: `STEPS=1000000 sbatch cc_mag_rorqual.sh 3s_vs_4z` overrides the 2M budget. Output goes to
`mag_profile/slurm-mag-<jobid>.out`; wandb offline runs land under `<repo>/wandb/`.

### Submit (Narval)
Edit the `REPO`, `VENV`, `SC2PATH` and `--account` lines in `cc_mag_narval.sh`, and confirm the MIG
gres name once with `sinfo -o "%G" | sort -u` (expected `a100_3g.20gb`). Then:

```bash
cd mag_profile
sbatch cc_mag_narval.sh 3s_vs_4z                                   # MAG
sbatch cc_mag_narval.sh 3s_vs_4z /home/chidamv/projects/def-dpmeger/chidamv/three-baselines/MAG_2      # MAMBA
sbatch cc_mag_narval.sh 3s_vs_4z /home/chidamv/projects/def-dpmeger/chidamv/three-baselines/MABL/mabl  # MABL
```

### Before scaling out: the two checks that still need a GPU
1. **Profiling job (gate F2)** — `sbatch cc_mag_profile_gpu.sh` from `mag_profile/` (~30 min). It
   times one learner update per phase for all three repos and prints the peak GPU memory per phase.
   The actor-phase peak must be under 20 GB for the Narval slice.
2. **One 100k-step run per algorithm on `3s_vs_4z`** (`STEPS=100000 sbatch ...`) to confirm the win
   rate is still in the range of the published curves before launching the 2M-step sweeps.

### Measured on Narval (A100 3g.20gb, 8 cores, 4 workers), projected to 2M steps
From the colleague's runs of 2026-09-07 (last-quarter throughput; ±30%):

| map | MAG | MABL |
|---|---|---|
| 3s_vs_4z | 5.7 h | 6.7 h |
| 3s_vs_5z | 4.8 h | 7.1 h |
| corridor | 6.0 h | 9.5 h |
| 2c_vs_64zg | 6.0 h | 11.5 h |
| 6h_vs_8z | 6.8 h | 11.3 h |
| 3s5z_vs_3s6z | 8.2 h | 10.2 h |
| MMM2 | 9.0 h | 11.9 h |
| so_many_baneling | 9.2 h | 12.3 h |
| 27m_vs_30m | crashes at start on a 20 GB slice (see Notes) | same |

MAG fits `--time=11:59:00` on every map above. MABL is slower despite its cheaper learner because it
is environment-bound (one StarCraft II process, one worker); on the five maps over ~10 h use a smaller
`STEPS=` or give MABL parallel workers. Driver RSS was 1.6–3.6 GB in all runs and `--mem=36G` was ample.

### Notes
- **If a run stops early with no error in the `.out` file**, look at the `.err` file: a wandb service
  crash shows up there as `Fatal Python error` followed by `BrokenPipeError` in `wandb.log`. The
  hardening above prevents this; runs started before it must be relaunched with the updated code.
- **27m_vs_30m** (27 agents) needs ~13 GB for the world-model phase alone at `MODEL_BATCH_SIZE=120`;
  run it with `MODEL_BATCH_SIZE=40, MODEL_EPOCHS=60` or on a full GPU.
- Seeds: MAG and MAMBA draw a random seed per run (`train.py`); MABL fixes `RANDOM_SEED = 23` — edit
  it for multi-seed studies.
- MABL is single-process; `--n_workers` is accepted but inert, so 4 cores are enough for it.
- Always pass `--env=starcraft`; the parser's default is `flatland`, whose code path is disabled.
- Progress is printed every 20 episodes (`map, cur_step, incre_win_rate`); with wandb enabled the
  per-update losses and win rate are logged as well.
