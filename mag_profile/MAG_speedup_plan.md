# Plan: speeding up MAG (causes 2, 3, 4) — code changes are allowed

Baseline (A100, 3s_vs_4z, per episode of ~70 env steps): 37 s = model 11 s + m_r 9 s + actor/MPC 16 s.
Order of work: exact changes first (steps A, B), then a mild semantic change (C), then the
update-frequency trade-off (D). Each step has its own verification so regressions are caught early.

## Step A — cause 4: train the m_r predictor on losses the model phase already computes (exact)
Files: `agent/learners/DreamerLearner.py` (`step`, `train_model`, `train_m_r_predictor`),
`agent/optim/loss.py` (`model_loss`, `m_r_perdictor_loss`, `get_model_loss_for_m_r_training`).
1. Make `model_loss` also return the per-step loss *with* the `dis` term and the detached prior
   features `[prior.stoch, deter]` (what `m_r_perdictor_loss` feeds the predictor). Both are already
   computed inside `model_loss`; today they are thrown away (`loss, _ = model_loss(...)`).
2. In `step`, interleave: after each of the 60 `train_model` epochs, call the predictor update on that
   epoch's `(features, per_step_loss)`. Delete the separate 60-epoch m_r loop and both extra model
   rollouts (`get_model_loss_for_m_r_training` and the `rollout_representation` inside
   `m_r_perdictor_loss`). Predictor epochs stay 60; the only behaviour difference is that labels come
   from the model as it trains rather than from the final model of the phase (the unused
   `DreamerMemory.sample_all` shows this was the original design).
Saving: ~9 s -> <0.5 s per episode (about 25% of the cycle).
Verify: fixed-seed synthetic buffer (profile_learner.py): predictor loss falls 1.5 -> ~0.05 as in the
wandb logs; timing table shows phase 2 gone.

## Step B — cause 3, part 1: remove host syncs and list churn from the MPC loop (exact)
File: `networks/dreamer/rnns.py` (`MPCPredict`, `rollout_policy`, `RSSMTransition.para_predict`).
1. `traj_losses` stays a GPU tensor `(n_trajs,)`; accumulate with tensor ops; select with
   `torch.argmin` and index the stacked first-step prediction with that tensor — one sync per
   imagination step instead of one per planning step (6x fewer), and drop the `np.random.choice`
   bookkeeping (`ranlosses`/`minlosses`) that is only used for a commented print.
2. Replace `pi[avail_actions == 0] = -1e10` with `pi = pi.masked_fill(avail_actions == 0, -1e10)`.
3. `para_predict` already returns stacked `logits/stoch/deter`; use them directly instead of
   rebuilding a Python list of `RSSMState`s and re-stacking `get_features()` every planning step.
   Keep only the final `first_pred[best]` selection as a gather on the stacked tensors.
Saving: several seconds per episode; larger on allocations like the Sep 6 node (its 1.9x slowdown was
confined to this sync-heavy phase).
Verify: bit-identical actor/critic losses vs the original on a fixed seed (these changes do not alter
the math), then timing.

## Step C — cause 3, part 2: run the 4 actor epochs as one 4x-batched imagination rollout (mild change)
File: `agent/learners/DreamerLearner.py` (`step`, `train_agent`).
1. Sample one batch of `EPOCHS*BATCH_SIZE` (=160) sequences and call `actor_rollout` once. The 90
   sequential MPC transition passes are latency-bound, so a 4x wider batch costs roughly the same per
   step: ~4x fewer sequential passes.
2. PPO: keep the total number of gradient steps the same (today 4 x 5 x 6 = 120 minibatches of
   <=2000 rows; batched it is 5 x 22 = 110 — raise `PPO_EPOCHS` to 6 or shrink the chunk to keep parity).
Semantic difference: today each actor epoch re-imagines with the actor updated by the previous epoch;
batched, all imagination uses the start-of-episode actor. This is a standard trade and is fine for an
adapted baseline; note it in the run log.
Saving: actor phase 16 s -> ~5-6 s.
Verify: timing; a 100k-step 3s_vs_4z run reaching the paper's MAG win rate (~0.6-0.8 by 100k).

## Step D — cause 2: update less often, and decide the epoch counts by experiment
Files: `configs/dreamer/DreamerLearnerConfig.py` only (`N_SAMPLES`, `MODEL_EPOCHS`,
`m_r_predictor_epochs`, `EPOCHS`).
1. The accumulation path already exists (`accum_samples < N_SAMPLES -> return`). Set `N_SAMPLES` in
   transitions (e.g. 250, 500) so one learner cycle serves ~4-8 episodes instead of 1. Cost per env step
   falls linearly; sample efficiency may fall too, because the update-to-data ratio drops from ~0.85
   model updates per env step to ~0.1-0.2.
2. Run the sweep at the paper budget (100k steps, 3s_vs_4z, 2-3 seeds): `N_SAMPLES in {1 episode,
   250, 500}` x `MODEL_EPOCHS in {60, 30}`. Pick the cheapest setting whose 100k win rate is within
   noise of the baseline curve. After step A the `m_r` epochs are almost free, so leave them at 60.
3. Optional knobs if more speed is needed (each changes the algorithm; cost is linear in each):
   `MPCHorizon` 6 -> 4, `n_trajs` 4 -> 3, `rollout_max_length` 15 -> 10, or `use_epsilon_MPC`
   (MPC only on a fraction of imagination steps).

## Expected outcome (A100, per learner cycle)
| after step | cycle | 1M steps at ~70 steps/episode |
|---|---|---|
| baseline | 37 s | ~6 days |
| A | ~28 s | ~4.5 days |
| A+B+C | ~17 s | ~2.8 days |
| A+B+C+D (N_SAMPLES=500) | ~17 s per ~7 episodes | ~10 h |

## Applies to the other two baselines?
Step A is MAG-only. Step B's masked-fill is harmless everywhere. Step C (batched imagination) and
step D apply unchanged to MAG_2/MAMBA (same code) and, with the single-worker runner, to MABL.

## Tooling already in place
`/scratch/zwang182/three-baselines-valliappan/mag_profile/profile_learner.py` times one `learner.step()` per phase on synthetic
data (CPU or GPU; needs `PYTHONPATH=<repo>:/scratch/zwang182/three-baselines-valliappan/mag_profile/stubs` on MARLenv). Use it
before/after every step; `cc_mag_profile_gpu.sh` runs it on an H100.
