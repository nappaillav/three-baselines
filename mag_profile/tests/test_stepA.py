"""Step A exactness: new model_loss label/input vs old get_model_loss_for_m_r_training / m_r_perdictor_loss input."""
import sys, numpy as np, torch, ray
torch.set_num_threads(4)
from configs.EnvConfigs import StarCraftConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from environments import Env
import agent.learners.DreamerLearner as L
from agent.optim import loss as lm
from networks.dreamer.rnns import rollout_representation
env = StarCraftConfig("3s_vs_4z").create_env()
cfg = DreamerLearnerConfig(); cfg.IN_DIM, cfg.ACTION_SIZE, cfg.n_ags = env.n_obs, env.n_actions, env.n_agents; n = env.n_agents; env.close()
cfg.ENV_TYPE = Env.STARCRAFT; cfg.DEVICE = 'cpu'; cfg.use_wandb = False
learner = L.DreamerLearner(cfg)
np.random.seed(0)
for _ in range(20):
    T = 60
    obs = np.random.randn(T, n, cfg.IN_DIM).astype(np.float32)
    act = np.eye(cfg.ACTION_SIZE, dtype=np.float32)[np.random.randint(0, cfg.ACTION_SIZE, (T, n))]
    d = np.zeros((T, n, 1), np.float32); d[-1] = 1
    learner.replay_buffer.append(obs, act, np.random.rand(T, n, 1).astype(np.float32), d, np.zeros_like(d), d.copy(), np.ones((T, n, cfg.ACTION_SIZE), np.float32))
learner.replay_buffer.init_sampled_idx()
s = learner.replay_buffer.sample(40)
model = learner.model[0].eval()
args = (s['observation'], s['action'], s['av_action'], s['reward'], s['done'], s['fake'], s['last'])
with torch.no_grad():
    torch.manual_seed(123); _, old_label = lm.get_model_loss_for_m_r_training(cfg, model, *args)
    torch.manual_seed(123); _, new_label, new_input = lm.model_loss(cfg, model, *args)
    # old predictor input, as computed inside m_r_perdictor_loss
    torch.manual_seed(123)
    obs = s['observation']; ts, bs = obs.shape[:2]
    embed = model.observation_encoder(obs.reshape(-1, n, obs.shape[-1])).reshape(ts, bs, n, -1)
    prev = model.representation.initial_state(bs, n, device=obs.device)
    old_input = rollout_representation(model.representation, ts, embed, s['action'], prev, s['last'])[0].get_features()
print("label shapes", tuple(old_label.shape), tuple(new_label.shape), "| input shapes", tuple(old_input.shape), tuple(new_input.shape))
print("label max|diff|", (old_label - new_label).abs().max().item(), "allclose:", torch.allclose(old_label, new_label, atol=1e-5, rtol=1e-4))
print("input max|diff|", (old_input - new_input).abs().max().item(), "allclose:", torch.allclose(old_input, new_input, atol=1e-6))
print("label mean", new_label.mean().item(), "| dis contributes on steps 1: ->", (new_label[1:] - (new_label[1:] - 0)).shape)
assert torch.allclose(old_label, new_label, atol=1e-5, rtol=1e-4) and torch.allclose(old_input, new_input, atol=1e-6)
print("STEP A EXACTNESS: PASS")
