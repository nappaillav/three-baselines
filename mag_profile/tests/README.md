One-off exactness tests used while implementing steps A and B (run from the MAG dir with
`PYTHONPATH=$PWD:../mag_profile/stubs`, MARLenv activated):
- `test_stepA.py` compares the new `model_loss` label/input with the OLD `get_model_loss_for_m_r_training`
  / `m_r_perdictor_loss` path. The old functions were deleted after it passed, so to re-run it check out
  `MAG/agent/optim/loss.py` from commit 36f4fd1 (pre-A) into a scratch copy, or read the recorded output
  in IMPLEMENTATION_REPORT.md.
- `test_stepB.py <path to rnns_old.py>` compares `rollout_policy` of the saved pre-B `rnns_old.py`
  (included here) with the current one under a fixed seed (torch.equal). Re-runnable as is.
`../smoke_train.py`: bounded CPU end-to-end run of `train.py` (SMOKE_EPISODES episodes, DEVICE forced to cpu)
without editing the repo; run it from inside a repo dir with SC2PATH exported.
- `test_item1.py <profile_learner.py> <map> <use_mpc> <model_epochs> <mr_epochs> <epochs> [threads] [device]`
  checks the chunked `critic_rollout` against a verbatim copy of the pre-change whole-batch version.
  `MODE=verify` (default) runs both at the call site under deterministic sampling and prints PASS/FAIL;
  `MODE=ref` swaps in the pre-change version and `MODE=new` uses the repo code, for A/B timing and
  memory. Re-runnable as is, from any of the three repo directories. Example:
  `cd MABL/mabl && MODE=verify BATCH_SIZE=10 MODEL_BATCH_SIZE=8 PYTHONPATH=$PWD:../../mag_profile/stubs \
   python ../../mag_profile/tests/test_item1.py ../../mag_profile/profile_learner.py 27m_vs_30m 0 1 1 1 4 cpu`
