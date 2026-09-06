# Three model-based MARL baselines — setup information

Written 2026-08-31 on Rorqual (Digital Research Alliance of Canada). Everything in this note was
verified by actually running all three repositories end-to-end (environment creation, StarCraft II
launch, rollout collection, and at least one full learner update each) under a Python 3.10
virtualenv on Rorqual on 2026-08-31. If you follow this note on another Alliance / Compute Canada
cluster you should end up with an equivalent working setup.

---

## 1. What is in this folder

| folder | algorithm | paper | entry point |
|---|---|---|---|
| `MAG/` | **MAG** | "Models as Agents: Optimizing Multi-Step Predictions of Interactive Local Models in Model-Based Multi-Agent Reinforcement Learning" (AAAI 2023) | `MAG/train.py` |
| `MAG_2/` | **MAMBA** | the MAMBA baseline | `MAG_2/train.py` |
| `MABL/mabl/` | **MABL** | "MABL: Bi-Level Latent-Variable World Model for Sample-Efficient Multi-Agent Reinforcement Learning" | `MABL/mabl/train.py` (note the extra `mabl/` level) |

All three share one MAMBA-derived, Dreamer-style codebase (world model + PPO-style learner, SMAC
rollout workers), so their structure is nearly identical (`agent/`, `configs/`, `env/`,
`networks/`, `train.py`).

Facts worth knowing before you touch anything:

- **MAG and MAG_2 differ in exactly one code line**: `configs/dreamer/DreamerLearnerConfig.py`
  sets `use_MPCmodel = True` in MAG and `False` in MAG_2. Everything else (apart from shell
  scripts) is byte-identical. So MAG = MAMBA + the MPC model.
- **MAG / MAG_2 are multi-process via ray**: `agent/runners/DreamerRunner.py` starts a
  `DreamerServer` with `--n_workers` ray actors, one StarCraft II process per worker.
- **MABL is single-process**: its `@ray.remote` decoration is commented out and the runner drives
  one in-process worker. The `--n_workers` argument is accepted but does nothing. (`import ray`
  still executes, so ray must still be installed.)
- The `cc_*.sh`, `run.sh`, `run2.sh` Slurm scripts included in each repo are from previous
  clusters and are **stale**: they reference a venv that no longer exists (`MBVDEnv`),
  cluster-specific GPU/MIG names, and old account lines. Use them only as templates.
- The `requirements.txt` files **inside** each repo (numpy 1.18 / torch 1.7 / ray 1.5) are stale
  too. Ignore them; use the `requirements.txt` next to this note.

---

## 2. We only run SMAC — keep the Flatland code disabled

The codebase originally supports two benchmarks, SMAC and Flatland. **We only use SMAC.**
`flatland-rl` is not installed in the venv and must not be installed. That is safe because the
Flatland code path is already disconnected — but only as long as the following stay as they are:

- `configs/EnvConfigs.py` (all three repos): the two Flatland imports near the top
  (`from env.flatland.Flatland import ...`, `from env.flatland.GreedyFlatland import ...`) are
  **commented out**. Keep them commented.
- `configs/__init__.py` (all three repos): the `configs.flatland.*` imports are commented out.
  Keep them commented.
- `agent/workers/DreamerWorker.py` (all three repos): the
  `from flatland.envs.agent_utils import RailAgentStatus` import is commented out. Keep it
  commented.

Two traps that follow from this:

1. **`--env` defaults to `"flatland"`** in `train.py`'s argument parser (all three repos).
   You must **always pass `--env=starcraft`** on the command line. Running without it (or with
   `--env=flatland`) hits the Flatland branch, which references names that are no longer imported,
   and crashes.
2. The `FlatlandConfig` classes and `env/flatland/` directory still exist as dead code. Leave
   them alone; they are harmless unless instantiated.

---

## 3. Setting up the Python venv (Alliance / Compute Canada cluster)

Standard Alliance practice: module Python + `virtualenv` + the shared wheelhouse. No conda.
Do all of this on a **login node** (compute nodes have no internet).

```bash
module purge
module load StdEnv/2023
module load python/3.10 cuda/12.2

virtualenv --no-download /home/chidamv/env/mbmarlEnv          # put it in HOME or PROJECT, NEVER in scratch
source /home/chidamv/env/mbmarlEnv/bin/activate
pip install --no-index --upgrade pip
```

(`scipy-stack` is not needed: numpy and matplotlib are installed into the venv below.)

Then install the dependencies in three steps, matching the three groups in `requirements.txt`:

```bash
# Group 1 -- from the Alliance wheelhouse (cluster-tuned wheels):
pip install --no-index numpy==1.25.2 torch==2.5.0 ray==2.2.0 matplotlib==3.7.2 \
    more-itertools==9.1.0 setproctitle==1.3.4 pygame==2.5.2 protobuf==3.20.3 \
    absl-py==1.4.0 grpcio==1.51.3

# Group 2 -- from PyPI (these are not in the wheelhouse; login node has internet):
pip install PySC2==4.0.0 s2clientprotocol==5.0.14.93333.0

# Group 3 -- oxwhirl SMAC, pinned to the verified commit:
pip install "SMAC @ git+https://github.com/oxwhirl/smac.git@12614f1760427026cce82083dc3f0ab3ff1d939e"
```

**Critical warning about SMAC:** the name `smac` on both PyPI and the Alliance wheelhouse resolves
to *SMAC3*, an unrelated AutoML library. Never `pip install smac` or `pip install --no-index smac`.
Only the git URL above installs the StarCraft Multi-Agent Challenge package these repos need.

Notes on the pins:
- These are the exact versions the three repos were verified with. The wheelhouse is shared
  across Alliance clusters (CVMFS), so the same `+computecanada` wheels should be available
  everywhere under `StdEnv/2023` + `python/3.10`. If a pinned wheelhouse version is missing on
  your cluster, check `avail_wheels <package> --python 3.10` and take the nearest version.
- ray 2.2.0 is far newer than the repos' internal pin (1.5.2); the repos only use the stable
  `ray.init/remote/wait/get` API and were verified working on 2.2.0.
- `setproctitle` is imported by `agent/learners/DreamerLearner.py` in MAG/MAG_2. It happens to
  work even without installing it, because ray vendors a copy that becomes importable after
  `import ray` — but do not rely on that; install it explicitly as above.
- `pip install --no-index -r requirements.txt` as a single command will NOT work (groups 2 and 3
  are not in the wheelhouse). Follow the three steps.

Quick validation (login node, no GPU needed):

```bash
python -c "
import ray, torch, numpy, matplotlib, more_itertools, setproctitle
from smac.env import StarCraft2Env
print('torch', torch.__version__, '| ray', ray.__version__, '| numpy', numpy.__version__)
print('CUDA arch flags:', torch._C._cuda_getArchFlags())"
```

(Use `torch._C._cuda_getArchFlags()`, not `torch.cuda.get_arch_list()` — the latter returns `[]`
on any node without a visible GPU and looks like a broken install when it isn't.)

---

## 4. StarCraft II and the SMAC maps

The repos need a Linux StarCraft II installation, **version 4.10 (Base75689)**. Results are not
comparable across SC2 versions, and our other experiments all use 4.10 — install exactly that one.

1. Download SC2 4.10 on a login node:
   `http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip` (unzip password: `iagreetotheeula`).
   Put it somewhere permanent, e.g. `~/MARL/StarCraftII/`.
2. Install the SMAC maps: a copy of `SMAC_Maps.zip` ships in this folder at
   `MABL/mabl/Starcraft/SMAC_Maps.zip`. The zip already contains a top-level `SMAC_Maps/`
   directory, so unzip it into `$SC2PATH/Maps/` (create the `Maps` directory if needed),
   which yields `$SC2PATH/Maps/SMAC_Maps/*.SC2Map`. The `__MACOSX/` folder it also extracts
   can be deleted.
3. **Export `SC2PATH` in every job script**:
   `export SC2PATH=/path/to/StarCraftII`.
   Do not rely on the defaults in the code: `MAG/train.py` and `MAG_2/train.py` hardcode a
   Rorqual-specific fallback path, and `MABL/mabl/train.py` sets no default at all.

---

## 5. Running the baselines

```bash
# MAG (from inside MAG/):
python train.py --env=starcraft --env_name=3m --n_workers=2
# MAG_2 / MAMBA (from inside MAG_2/): same command
# MABL (from inside MABL/mabl/): same command (--n_workers is inert, see section 1)
```

Run from inside the repo directory (imports are relative to it), or set `PYTHONPATH` to the repo
root. `--env_name` is the SMAC map name (`3m`, `2s_vs_1sc`, `3s_vs_3z`, `8m_vs_9m`, ...).

Behavior to be aware of:

- **Device**: `DEVICE = 'cuda'` is hardcoded in `configs/dreamer/DreamerAgentConfig.py` (learner)
  and `configs/dreamer/DreamerControllerConfig.py` (controller) in all three repos. For a CPU run,
  change both to `'cpu'`. torch 2.5.0 from the wheelhouse has H100 (sm_90) kernels, so H100/MIG
  GPU jobs work.
- **Learning start**: the learner buffers transitions and does nothing until
  `MIN_BUFFER_SIZE = 500` transitions have accumulated (a dozen-plus episodes on small maps).
  A run that "only collects episodes" at the start is normal.
- **Seeds**: MAG and MAG_2 draw a random seed each run (`train.py`); MABL fixes `RANDOM_SEED = 23`.
  Edit `train.py` if you need controlled seeds.
- **Logging without wandb**: progress goes to stdout only — a
  `map: <name>, cur_step: <n>, incre_win_rate: <r>` line every 20 episodes. For persistent
  metrics, enable wandb (section 6).
- **No checkpointing**: none of the three repos saves models or supports resume. A run must fit
  inside its wall-time; budget `--time` accordingly.
- **Run length**: `train.py` sets `steps=int(1e6)` in MAG/MAG_2 (MABL: `steps=10**10`,
  `episodes=50000`); edit the `Experiment(...)` call in `train.py` to change budgets.
- Compute nodes have **no internet**: everything (SC2, maps, wheels, wandb in offline mode) must
  be on disk before the job starts.
- Cosmetic: at interpreter exit, pysc2 often prints
  `AttributeError: __enter__` from `sc_process.py`'s `__del__`. This happens after training has
  finished and is harmless — do not chase it.

Resource sizing observed in our verification runs (CPU, map `3m`): MAG's MPC path makes it ~7x
slower per learner update than MAG_2/MABL. Treat that as a hint, not a measurement — profile a
short job on your cluster before sizing the real ones.

---

## 6. Enabling wandb logging

wandb support is scaffolded in all repos but disabled. Three kinds of change are needed.

### 6.1 Install wandb into the venv

```bash
pip install --no-index wandb     # it is in the Alliance wheelhouse
```

### 6.2 Code changes — MAG and MAG_2 (identical edits, 4 files each)

1. `train.py` lines 5–7: uncomment `import socket`, `import setproctitle`, `import wandb`.
   In the existing `wandb.init(...)` block (~line 99), fill in `project=''` with a real project
   name; fill in or delete the `entity=''` argument.
2. `agent/runners/DreamerRunner.py` line 2: uncomment `import wandb` (otherwise the `wandb.log`
   calls at lines ~49/53 are a NameError at the first log interval).
3. `agent/learners/DreamerLearner.py`: add `import wandb` at the top of the file (only a
   commented-out import exists inside `__init__`, ~lines 88–89). This enables the `m_r_loss`
   (MAG only) and `Agent/val_loss` / `Agent/actor_loss` logs.
4. `configs/dreamer/DreamerAgentConfig.py` line 39: set `self.use_wandb = True`. This single flag
   gates every wandb call.

### 6.3 Code changes — MABL (3 files)

1. `MABL/mabl/train.py` has **no wandb init at all** (the whole block was stripped). Copy the
   init block from `MAG/train.py` (the `wandb.init(...)` call plus the three
   `wandb.define_metric` lines and the imports) into it, before `train_dreamer(...)`.
   Do not use the alternative init commented inside `DreamerLearner.py` — it hardcodes a project
   name.
2. `MABL/mabl/agent/runners/DreamerRunner.py` line 2: uncomment `import wandb`.
   **Do NOT uncomment line 3** — it is a `wandb.login(key=...)` with a hardcoded API key.
   Delete that line, and if that key was ever real, revoke it in the wandb account settings.
3. `MABL/mabl/configs/dreamer/DreamerAgentConfig.py` line 47: set `self.use_wandb = True`.
4. Optional: MABL only logs `incre_win_rate` by default. For loss curves, also uncomment the
   `wandb.log` lines in `MABL/mabl/agent/learners/DreamerLearner.py` (~lines 133 and 147) and add
   `import wandb` at the top of that file.

### 6.4 Cluster operation: offline mode

Compute nodes have no internet, so wandb must run offline and be synced afterwards:

```bash
# in the job script:
export WANDB_MODE=offline
# LOG_FOLDER is 'wandb/' RELATIVE TO THE CWD in all three configs
# (configs/dreamer/DreamerAgentConfig.py). Point it somewhere absolute with quota to spare,
# either by editing the config or by cd-ing to a suitable working dir, and create it first:
mkdir -p <log_dir>
```

After the job, on a login node:

```bash
wandb login          # once
wandb sync <log_dir>/wandb/offline-run-*
```

Multiprocess safety is a non-issue: only the main process (runner + learner) calls `wandb.log`;
the ray workers never touch wandb.

---

## 7. Suggested first test on the new cluster

1. Venv validation snippet from section 3 (login node, seconds).
2. Short CPU sanity run on the login node if your cluster's policy allows it, or as a small test
   job: set both `DEVICE` fields to `'cpu'`, and in `configs/dreamer/DreamerLearnerConfig.py`
   temporarily shrink `MIN_BUFFER_SIZE` (e.g. 60) and `MODEL_EPOCHS` (e.g. 2) so a learner update
   actually executes within a few episodes. Run
   `python train.py --env=starcraft --env_name=3m --n_workers=1` and wait for the first
   `incre_win_rate` print or clean exit. Revert the config changes afterwards.
3. Then a real GPU job with the defaults.
