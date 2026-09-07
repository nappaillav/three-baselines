"""Step B exactness: imagined rollout identical (torch.equal) between the old rnns.py (saved copy) and the new one."""
import sys, types, importlib.util, numpy as np, torch, ray
torch.set_num_threads(8)
from configs.EnvConfigs import StarCraftConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from environments import Env
import agent.learners.DreamerLearner as L
import networks.dreamer.rnns as rnns_new
spec = importlib.util.spec_from_file_location("rnns_old", sys.argv[1]); rnns_old = importlib.util.module_from_spec(spec); spec.loader.exec_module(rnns_old)
env = StarCraftConfig("3s_vs_4z").create_env()
cfg = DreamerLearnerConfig(); cfg.IN_DIM, cfg.ACTION_SIZE, cfg.n_ags = env.n_obs, env.n_actions, env.n_agents; n = env.n_agents; env.close()
cfg.ENV_TYPE = Env.STARCRAFT; cfg.DEVICE = 'cpu'; cfg.use_wandb = False
learner = L.DreamerLearner(cfg)
model, actor, mr = learner.model[0].eval(), learner.actor, learner.m_r_predictor
model.transition.para_predict = types.MethodType(rnns_old.RSSMTransition.para_predict, model.transition)   # old path needs it
np.random.seed(0)
for _ in range(20):
    T = 60
    obs = np.random.randn(T, n, cfg.IN_DIM).astype(np.float32)
    act = np.eye(cfg.ACTION_SIZE, dtype=np.float32)[np.random.randint(0, cfg.ACTION_SIZE, (T, n))]
    d = np.zeros((T, n, 1), np.float32); d[-1] = 1
    learner.replay_buffer.append(obs, act, np.random.rand(T, n, 1).astype(np.float32), d, np.zeros_like(d), d.copy(), np.ones((T, n, cfg.ACTION_SIZE), np.float32))
learner.replay_buffer.init_sampled_idx()
s = learner.replay_buffer.sample(40)
obs = s['observation']; T, B = obs.shape[:2]
with torch.no_grad():
    embed = model.observation_encoder(obs.reshape(-1, n, obs.shape[-1])).reshape(T, B, n, -1)
    prev = model.representation.initial_state(B, n, device=obs.device)
    torch.manual_seed(1); prior, post, _ = rnns_new.rollout_representation(model.representation, T, embed, s['action'], prev, s['last'])
    post = post.map(lambda x: x.reshape((T - 1) * B, n, -1))
    for (steps, H, K) in [(5, 3, 2), (15, 6, 4)]:
        cfg.MPCHorizon, cfg.n_trajs = H, K
        torch.manual_seed(7); old = rnns_old.rollout_policy(mr, model.observation_decoder, model.transition, model.av_action, steps, actor, post, cfg)
        torch.manual_seed(7); new = rnns_new.rollout_policy(mr, model.observation_decoder, model.transition, model.av_action, steps, actor, post, cfg)
        checks = {"stoch": torch.equal(old["imag_states"].stoch, new["imag_states"].stoch),
                  "deter": torch.equal(old["imag_states"].deter, new["imag_states"].deter),
                  "logits": torch.equal(old["imag_states"].logits, new["imag_states"].logits),
                  "actions": torch.equal(old["actions"], new["actions"]),
                  "av_actions": torch.equal(old["av_actions"], new["av_actions"]),
                  "old_policy": torch.equal(old["old_policy"], new["old_policy"])}
        print(f"steps={steps} H={H} K={K}: shapes {tuple(new['imag_states'].stoch.shape)} ->", checks)
        assert all(checks.values())
print("STEP B EXACTNESS: PASS (torch.equal on all imagined tensors)")
