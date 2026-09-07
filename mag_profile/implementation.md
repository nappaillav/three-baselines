# Implementation plan — MAG speed-up (steps A–D + job setup)

Target repo: `/scratch/zwang182/three-baselines-valliappan/MAG` (MAG_2/MABL noted where applicable).
Decisions this implements: `DECISIONS.md`. Rationale and measurements: `MAG_slow_runtime_summary.md`,
`MAG_speedup_plan.md`. Line numbers refer to the fork as of 2026-09-06.

Work on a git branch (`git checkout -b speedup`), one commit per step, so each step can be timed and
bisected independently. Run the profiler (Step 0) before and after every step.

---

## Step 0 — Instrument first (no behaviour change)

**0.1 Profiler: per-phase peak GPU memory.** In `mag_profile/profile_learner.py`, extend `wrap()` so
that when `cfg.DEVICE == 'cuda'` each wrapped call does `torch.cuda.reset_peak_memory_stats()`
before and records `torch.cuda.max_memory_allocated()` after, printed as an extra column. This is what
closes the open item "does the actor phase fit in 20 GB".

**0.2 Profiler: accept the new hyperparameters.** Add optional environment-variable overrides
(`MODEL_BATCH_SIZE`, `ROLLOUT_LEN`, `MPC_H`, `N_TRAJS`, `PPO_MINIBATCH`) applied to `cfg` after
construction, so the same script measures the shipped and the decided configuration.

**0.3 Baseline numbers.** Run the profiler on the GPU job (`cc_mag_profile_gpu.sh`) with the shipped
config and record: phase times and peak memory. These are the reference for every later step.

---

## Step A — Predictor trained on model-phase losses (cause 4, exact)

Files: `agent/optim/loss.py`, `agent/learners/DreamerLearner.py`.

**A.1 `loss.py::model_loss` (lines 71–105)** — return the predictor's inputs and labels.
- The predictor label used today (`get_model_loss_for_m_r_training`, lines 29–69) is
  `rec + rew + pcont + div + av` per step **plus** the `dis` term on steps `1:` (each gated by the
  `config.rec/rew/avl/pcont/dis` flags). `model_loss` already computes every one of these; its
  `model_loss_per_step` (line 103) only lacks `dis`. Add it:
  ```python
  dis_loss, dis_per_step = info_loss(i_feat[1:], model, action[1:-1], 1. - fake[1:-1].reshape(-1))
  model_loss_per_step = rec_loss_per_step + rew_loss_per_step + pcont_loss_per_step + div_per_step + av_loss_per_step
  model_loss_per_step[1:] += dis_per_step.reshape(time_steps - 2, batch_size, n_agents, 1)
  ```
  (apply the same `config.*` gating as lines 50–67 if you want the flags to keep working).
- The predictor input used today (`m_r_perdictor_loss`, line 19–23) is `prior.get_features()`
  = `cat(prior.stoch, prior.deter)` for steps `0..T-2` — `prior` as returned by
  `rollout_representation` is already sliced `[:-1]`. Add:
  ```python
  mr_input = prior.get_features().detach()          # (T-1, B, n_agents, FEAT)
  return model_loss, model_loss_per_step.detach(), mr_input
  ```
- Delete `get_model_loss_for_m_r_training` and the rollout inside `m_r_perdictor_loss`; replace the
  latter by a pure function of tensors:
  ```python
  def m_r_predictor_loss(m_r_predictor, mr_input, mr_label):
      n_agents = mr_input.shape[2]
      out = m_r_predictor(mr_input.reshape(-1, n_agents, mr_input.shape[-1]))
      return F.smooth_l1_loss(out, mr_label.reshape(-1, n_agents, 1))
  ```

**A.2 `DreamerLearner.train_model` (lines 187–201)** — unpack the third return value and return it:
`loss, per_step, mr_input = model_loss(...)`; return `np.array(losses), mr_input, per_step`.
(`n_nets == 1` is asserted elsewhere; keep the loop but only the single model's tensors are used.)

**A.3 `DreamerLearner.train_m_r_predictor` (lines 172–185)** — new signature
`train_m_r_predictor(self, mr_input, mr_label)`; body = the new `m_r_predictor_loss` + the existing
optimizer step. Keep the `wandb.log({'m_r_loss': ...})` so the wandb curve stays comparable.

**A.4 `DreamerLearner.step` (lines 121–140)** — interleave and delete the second loop:
```python
for i in range(self.config.MODEL_EPOCHS):
    samples = self.replay_buffer.sample(self.config.MODEL_BATCH_SIZE)
    loss, mr_input, mr_label = self.train_model(samples)
    losses.append(loss)
    if self.config.use_MPCmodel:
        for _ in range(self.config.m_r_updates_per_model_epoch):   # new config, default 1
            self.train_m_r_predictor(mr_input, mr_label)
```
Remove the `m_r_predictor_epochs` loop (lines 130–139). `m_r_predictor_epochs` becomes unused;
with `MODEL_EPOCHS=20` the predictor gets 20 updates per cycle (60 before) — raise
`m_r_updates_per_model_epoch` to 3 if its loss curve looks under-trained (each update is ~2 ms).

**A.5 Verify.**
- Correctness: on a fixed batch with an untrained model, compute the old label
  (`get_model_loss_for_m_r_training`) and the new `model_loss_per_step` from the same call and assert
  `allclose` (do this once before deleting the old function). Same for inputs vs `prior.get_features()`.
- Behaviour: profiler shows phase 2 (m_r) gone; predictor loss on the wandb `m_r_loss` curve still
  falls from ~1.5 to <0.1 within the first cycles (as in the colleague's logs).
- Timing target on A100: cycle 37 s -> ~28 s at the shipped config.

---

## Step B — MPC loop without host syncs (cause 3 part 1, exact)

File: `networks/dreamer/rnns.py` (`para_predict` 104–123, `rollout_policy` 176–231, `MPCPredict` 234–272).

**B.1 `RSSMTransition.para_predict`** — take flat tensors, return flat tensors, no Python lists:
```python
def para_predict_flat(self, prev_actions, stoch, deter, mask=None):
    # all inputs (n_trajs*B, n_agents, .)
    x = self._rnn_input_model(torch.cat([prev_actions, stoch], dim=-1))
    x = self._attention_stack(x, mask=mask) if self.config.use_attn else self._fc(x)
    NB, n = x.shape[:2]
    deter_state = self._cell(x.reshape(1, NB * n, -1), deter.reshape(1, NB * n, -1))[0].reshape(NB, n, -1)
    logits, stoch_state = self._stochastic_prior_model(deter_state)
    return logits, stoch_state, deter_state
```

**B.2 `MPCPredict`** — rewrite (same math, one selection per imagination step, no `.cpu()`):
```python
def MPCPredict(policy, m_r_predictor, action, state, transition_model, config, av_action):
    K = config.n_trajs; B, n = action.shape[:2]
    rep = lambda x: x.unsqueeze(0).expand(K, *x.shape).reshape(K * B, *x.shape[1:])
    act, stoch, deter = rep(action), rep(state.stoch), rep(state.deter)
    traj_losses = torch.zeros(K, device=action.device)
    for t in range(config.MPCHorizon):
        logits, stoch, deter = transition_model.para_predict_flat(act, stoch, deter)
        if t == 0:
            first = (logits, stoch, deter)
        feat = torch.cat([stoch, deter], -1)
        loss = m_r_predictor(feat).view(K, B, n, 1)
        if config.discount_MPC:
            loss = loss * config.MPCgamma ** t
        traj_losses = traj_losses + loss.mean(dim=(1, 2, 3))
        act, pi = policy(feat)
        if av_action is not None:
            avail = av_action(feat).sample()
            pi = pi.masked_fill(avail == 0, -1e10)
            act = (OneHot(pi.shape[-1]).transform(pi.argmax(-1, keepdim=True)) if config.DeterPolForMo
                   else OneHotCategorical(logits=pi).sample())
    best = traj_losses.argmin()                      # 0-d tensor; indexing below needs no .item()
    sel = lambda x: x.view(K, B, n, -1)[best]
    return RSSMState(logits=sel(first[0]), stoch=sel(first[1]), deter=sel(first[2]))
```
Return only the state; delete `minlosses`/`ranlosses` bookkeeping in `rollout_policy` (lines
191, 217–224) — they only feed a commented-out print.

**B.3 `rollout_policy` line 205** — `pi = pi.masked_fill(avail_actions == 0, -1e10)`. Same in
`agent/optim/loss.py::actor_loss` line 171 (`new_policy = new_policy.masked_fill(...)`) and in
`agent/controllers/DreamerController.py` line 73 (harmless, CPU).

**B.4 Verify.** With `torch.manual_seed` fixed and the same batch, the per-step chosen trajectory and
the returned state must be identical to the old implementation (the only RNG consumer removed is
`np.random.choice`, which is numpy, so torch sampling streams are unchanged). Compare
`items["imag_states"]` tensors with `torch.equal` before/after on CPU. Then profiler timing.

---

## Step C — One batched imagination rollout per cycle (cause 3 part 2)

File: `agent/learners/DreamerLearner.py` (`step` 143–145, `train_agent` 203–255).

**C.1 `step`** — replace the `EPOCHS` loop with one call:
```python
samples = self.replay_buffer.sample(self.config.EPOCHS * self.config.BATCH_SIZE)   # 4 x 40 = 160
self.train_agent(samples)
```
`EPOCHS` keeps its meaning as a multiplier so the config stays readable.

**C.2 `train_agent` PPO loop (lines 233–253)** — make the minibatch size a config value:
`step = self.config.PPO_MINIBATCH` (new, default 1000; was the literal `2000` on line 236).
Bookkeeping: after this step and Step D, `actions.shape[0]` = (rollout_len−1)·(T−1)·EPOCHS·B
= 4·19·160 = 12,160 rows -> 13 minibatches per PPO epoch, 65 actor+critic updates per cycle with
`PPO_EPOCHS=5`. The shipped code did 120 per episode; the number of updates per *env step* is a
deliberate part of the decided trade-off and is validated by the 100k gate, not by parity.

**C.3 `agent/optim/loss.py::actor_loss` lines 166–168** — the `obs_as_pol_in` branch reshapes with
`config.BATCH_SIZE`; it is not used (`obs_as_pol_in=False`) but make it use the actual batch
(`imag_states.shape[1]`) so it does not silently break if the flag is ever enabled.

**C.4 Verify.** Profiler: actor phase time and peak memory at batch 160 (the memory number is the
one the 20 GB Narval slice depends on). Sanity: actor/critic losses finite; entropy annealing still
runs (`self.entropy *= ...` is per minibatch, so fewer minibatches per cycle means slower annealing —
acceptable; note it).

---

## Step D — Hyperparameters (`configs/dreamer/DreamerLearnerConfig.py`)

Set:
```python
self.MODEL_BATCH_SIZE = 120        # was 40
self.MODEL_EPOCHS = 20             # was 60
self.N_SAMPLES = 500               # was 1  (transitions between learner cycles)
self.rollout_min_length = 5        # was 15 (both min and max: rollout length is constant)
self.rollout_max_length = 5
self.MPCHorizon = 3                # was 6
self.n_trajs = 2                   # was 4
self.PPO_MINIBATCH = 1000          # new (Step C)
self.m_r_updates_per_model_epoch = 1   # new (Step A)
self.MODEL_LR = 5e-4               # unchanged for now; consider 1e-3 if model loss plateaus with batch 120
```
`m_r_predictor_epochs`, `HORIZON`, `EPOCHS` (still a multiplier) — leave as they are.
`N_SAMPLES` is compared with `accum_samples`, which counts transitions
(`DreamerLearner.step` line 107), so 500 = 500 env steps regardless of episode length.

Same values apply to `MAG_2` (MAMBA) except the MPC ones, which it ignores.

---

## Step E — Driver, ray, and job scripts

**E.1 `train.py`**
- line 82: `torch.set_num_threads(2)` (decided: fixed at 2, not derived).
- line 83: keep `CUDA_VISIBLE_DEVICES` handling; on Slurm the GPU is already isolated, so the line is
  harmless.
- lines 94–96: read `SC2PATH` from the environment only (`os.environ["SC2PATH"]`), fail loudly if
  unset — the hardcoded path is the colleague's and will not exist on other clusters.

**E.2 `agent/runners/DreamerRunner.py` line 9** —
`ray.init(num_cpus=n_workers, object_store_memory=512 * 1024 ** 2, include_dashboard=False)`.
Without `object_store_memory`, ray sizes the store from the node's RAM (500 GB) and can exceed the job's
cgroup memory limit.

**E.3 Job scripts** (`mag_profile/cc_mag_narval.sh`, `mag_profile/cc_mag_rorqual.sh`):
```bash
# Narval                                   # Rorqual
#SBATCH --gpus=a100_3g.20gb:1              #SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=8                  #SBATCH --cpus-per-task=8
#SBATCH --mem=24G                          #SBATCH --mem=24G
#SBATCH --time=14:00:00                    #SBATCH --time=14:00:00   (est. 9-10 h + margin)
#SBATCH --account=<account>
module purge; module load StdEnv/2023 python/3.10 cuda/12.2
source <venv>/bin/activate
export SC2PATH=<path>/StarCraftII
export OMP_NUM_THREADS=1                   # ray workers stay single-threaded
export WANDB_MODE=offline
python train.py --env=starcraft --env_name=$MAP --n_workers=4
```
Verify the Narval MIG name (`sinfo -o "%G" | sort -u` shows the exact gres string) before use.
The 3g.20gb row's 20 GB must be confirmed by Step 0.1's peak-memory column at the decided config.

**E.4 Wall-time safety.** There is no checkpointing in these repos. `--time` must cover the whole
2M-step run; if the first full run shows >12 h, either split by adding a save/restore of
`learner.params()` + optimizer states (small addition in `DreamerLearner`) or lower `--time` risk by
running 1M-step halves only after checkpointing exists.

---

## Step F — Verification gates (in order)

| gate | when | pass criterion |
|---|---|---|
| F1 profiler, CPU login node | after each of A, B, C | phase table changes as predicted; A/B exactness checks pass |
| F2 profiler, GPU job | after B and after D | cycle ~8-9 s per 500 steps; actor-phase peak memory < 20 GB |
| F3 100k-step run, 3s_vs_4z, 2 seeds | after D | win rate within noise of the paper's MAG curve (~0.6-0.8 at 100k); `m_r_loss` and model loss curves healthy |
| F4 first 4-worker production job | first Narval/Rorqual job | wandb system stats: driver ~1 core, no core at 100% other than driver + 4 SC2; cycle time matches F2 |
| F5 one full 2M-step run | before scaling seeds | finishes inside `--time`; curve continues to improve past 100k |

If F3 fails, the first knobs to revisit (in order of least cost to speed): `m_r_updates_per_model_epoch`
1->3, `PPO_MINIBATCH` 1000->500, `N_SAMPLES` 500->250, `rollout_max_length` 5->8.

---

## Ordering and effort

| step | files | effort | risk |
|---|---|---|---|
| 0 | profiler | 1 h | none |
| A | loss.py, DreamerLearner.py | 2-3 h incl. exactness test | low (exact) |
| B | rnns.py (+2 masked_fill sites) | 2-3 h incl. equality test | low (exact) |
| C | DreamerLearner.py | 1 h | medium (training dynamics) |
| D | config | 10 min | medium (validated by F3) |
| E | train.py, DreamerRunner.py, job scripts | 1 h | low |
