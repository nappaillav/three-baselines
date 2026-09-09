"""Item 1 verification: the chunked `critic_rollout` must equal the pre-change whole-batch version.

MODE=verify (default): at the call site, run BOTH the repo's current `critic_rollout` and an inline copy
    of the pre-change one on the same arguments and the same module state, then compare. Sampling inside
    `model.transition` is made deterministic (argmax one-hot) so the two are directly comparable; with
    real sampling they differ only in the order of the draws.
MODE=ref:  replace `critic_rollout` with the pre-change whole-batch version (for A/B timing and memory).
MODE=new:  run the repo code unchanged.

usage (from a repo dir, MARLenv active, SC2PATH exported):
    PYTHONPATH=$PWD:../mag_profile/stubs MODE=verify \
        python ../mag_profile/tests/test_item1.py ../mag_profile/profile_learner.py <map> <use_mpc> 1 1 1 4 cpu
"""
import sys, os, runpy, resource, inspect
import ray                                  # first: vendors setproctitle
import torch
import torch.distributions as td

MODE = os.environ.get("MODE", "verify")
if MODE == "verify":                        # remove randomness from the transition model's latent sampling
    td.OneHotCategorical.sample = lambda self, sample_shape=torch.Size(): torch.nn.functional.one_hot(
        self.probs.argmax(-1), self.probs.shape[-1]).to(self.probs)

import agent.optim.loss as L
from agent.utils.params import FreezeParameters
from agent.optim.utils import compute_return

IS_MABL = "global_raw_states" in inspect.signature(L.critic_rollout).parameters


def ref_mag(model, critic, states, rew_states, actions, raw_states, config):
    with FreezeParameters([model, critic]):
        imag_reward = L.calculate_next_reward(model, actions, raw_states)
        imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        value = critic(states, actions)
        discount_arr = model.pcont(rew_states).mean
    return compute_return(imag_reward, value[:-1], discount_arr, bootstrap=value[-1],
                          lmbda=config.DISCOUNT_LAMBDA, gamma=config.GAMMA)


def ref_mabl(model, critic, agent_states, global_states, rew_states, actions, agent_raw_states,
             global_raw_states, config):
    with FreezeParameters([model, critic]):
        imag_reward = L.calculate_next_reward(model, actions, agent_raw_states, global_raw_states)
        imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        value = critic(agent_states, global_states, actions)
        discount_arr = model.pcont(rew_states).mean
    return compute_return(imag_reward, value[:-1], discount_arr, bootstrap=value[-1],
                          lmbda=config.DISCOUNT_LAMBDA, gamma=config.GAMMA)


reference = ref_mabl if IS_MABL else ref_mag
current = L.critic_rollout
results = []


def comparing(*args, **kwargs):
    new = current(*args, **kwargs)
    old = reference(*args, **kwargs)
    same_shape = tuple(new.shape) == tuple(old.shape) and new.dtype == old.dtype
    diff = float((new - old).abs().max().item()) if same_shape else float("nan")
    results.append((same_shape and torch.allclose(new, old, atol=1e-5, rtol=1e-4),
                    tuple(new.shape), tuple(old.shape), diff))
    return new


if MODE == "verify":
    L.critic_rollout = comparing
elif MODE == "ref":
    L.critic_rollout = reference

prof = sys.argv[1]
sys.argv = [prof] + sys.argv[2:]
runpy.run_path(prof, run_name="__main__")

print(f"\nRESULT mode={MODE} repo={'MABL' if IS_MABL else 'MAG'} "
      f"peak_rss_GB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.2f}")
if MODE == "verify":
    for ok, s_new, s_old, d in results:
        print(f"  critic_rollout: new{s_new} vs old{s_old}  max|diff|={d:.3e}  -> {'MATCH' if ok else 'MISMATCH'}")
    print("ITEM 1 EXACTNESS:", "PASS" if results and all(r[0] for r in results) else ("FAIL (never called)" if not results else "FAIL"))
