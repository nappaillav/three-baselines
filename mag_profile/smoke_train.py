"""Bounded end-to-end smoke run of train.py on CPU: SMOKE_EPISODES episodes, DEVICE forced to cpu, threads 8.
Nothing in the repo is edited; overrides are monkeypatches applied before train.py runs."""
import os, sys, runpy, time, math, ray, torch, numpy as np
sys.path.insert(0, os.getcwd())
n_ep = int(os.environ.get("SMOKE_EPISODES", "16"))
_orig_threads = torch.set_num_threads
torch.set_num_threads = lambda n: _orig_threads(8)          # train.py's fixed 2 would make a CPU smoke run needlessly slow
import importlib; E = importlib.import_module("configs.Experiment")   # configs/__init__ shadows the submodule name with the class
_oi = E.Experiment.__init__
E.Experiment.__init__ = lambda self, steps, episodes, *a, **k: _oi(self, steps, n_ep, *a, **k)
from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
for cls in (DreamerControllerConfig, DreamerLearnerConfig):
    def _mk(o):
        def i(self): o(self); self.DEVICE = "cpu"
        return i
    cls.__init__ = _mk(cls.__init__)
import agent.learners.DreamerLearner as L
_ml, _ta = L.model_loss, L.DreamerLearner.train_agent
def tm(*a, **k):   # capture the scalar model loss at the loss-function level (MAG returns a tuple, MABL a tensor)
    r = _ml(*a, **k); loss = r[0] if isinstance(r, tuple) else r
    tm.losses.append(float(loss.detach().item())); return r
tm.losses = []
def ta(self, s):
    t = time.time(); _ta(self, s); print(f"[smoke] train_agent done in {time.time()-t:.1f}s; model losses this cycle: first={tm.losses[0]:.3f} last={tm.losses[-1]:.3f} all_finite={all(math.isfinite(x) for x in tm.losses)}", flush=True); tm.losses.clear()
L.model_loss, L.DreamerLearner.train_agent = tm, ta
sys.argv = ["train.py", "--env=starcraft", "--env_name=3m", "--n_workers=1"]
t0 = time.time()
runpy.run_path("train.py", run_name="__main__")
print(f"[smoke] finished {n_ep} episodes in {time.time()-t0:.0f}s", flush=True)
