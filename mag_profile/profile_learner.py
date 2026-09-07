"""Profile one DreamerLearner.step() on synthetic buffer data (no SC2 needed).
usage (from a repo dir): PYTHONPATH=$PWD:<mag_profile>/stubs python profile_learner.py <map> <use_mpc 0|1> <MODEL_EPOCHS> <MR_EPOCHS> <EPOCHS> [threads] [device cpu|cuda]
Optional env-var overrides applied to the learner config after construction:
  MODEL_BATCH_SIZE, BATCH_SIZE, ROLLOUT_LEN (rollout_min/max_length), MPC_H (MPCHorizon), N_TRAJS (n_trajs),
  PPO_MINIBATCH, PPO_EPOCHS, MR_UPDATES (m_r_updates_per_model_epoch), N_SAMPLES
On cuda, a per-phase peak-memory column (torch.cuda.max_memory_allocated during the call) is printed.
"""
import os, sys, time, numpy as np, torch, ray  # ray first: it vendors setproctitle
map_name, use_mpc = sys.argv[1], sys.argv[2] == '1'
ME, MRE, EP = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
torch.set_num_threads(int(sys.argv[6]) if len(sys.argv) > 6 else 8)
device = sys.argv[7] if len(sys.argv) > 7 else 'cpu'
torch.manual_seed(0); np.random.seed(0)

from configs.EnvConfigs import StarCraftConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from environments import Env
import agent.learners.DreamerLearner as L
import agent.optim.loss as lossmod
import networks.dreamer.rnns as rnns

env = StarCraftConfig(map_name).create_env()          # builds env info without launching SC2
cfg = DreamerLearnerConfig()
cfg.IN_DIM, cfg.ACTION_SIZE, cfg.n_ags = env.n_obs, env.n_actions, env.n_agents
cfg.N_AGENTS = env.n_agents
if hasattr(env, 'n_state'):
    cfg.STATE_DIM = env.n_state
n_agents = env.n_agents
env.close()
cfg.ENV_TYPE = Env.STARCRAFT; cfg.DEVICE = device; cfg.use_wandb = False
cfg.use_MPCmodel = use_mpc
cfg.MODEL_EPOCHS, cfg.EPOCHS = ME, EP
if hasattr(cfg, 'm_r_predictor_epochs'):
    cfg.m_r_predictor_epochs = MRE
_ov = {'MODEL_BATCH_SIZE': 'MODEL_BATCH_SIZE', 'BATCH_SIZE': 'BATCH_SIZE', 'MPC_H': 'MPCHorizon', 'N_TRAJS': 'n_trajs',
       'PPO_MINIBATCH': 'PPO_MINIBATCH', 'PPO_EPOCHS': 'PPO_EPOCHS', 'MR_UPDATES': 'm_r_updates_per_model_epoch',
       'N_SAMPLES': 'N_SAMPLES'}
for k, attr in _ov.items():
    if k in os.environ:
        setattr(cfg, attr, int(os.environ[k]))
if 'ROLLOUT_LEN' in os.environ:
    if hasattr(cfg, 'rollout_max_length'):   # MAG / MAG_2
        cfg.rollout_min_length = cfg.rollout_max_length = int(os.environ['ROLLOUT_LEN'])
    else:                                   # MABL: imagination length is HORIZON
        cfg.HORIZON = int(os.environ['ROLLOUT_LEN'])
desc = {k: getattr(cfg, k, None) for k in ['MODEL_EPOCHS', 'm_r_predictor_epochs', 'm_r_updates_per_model_epoch', 'EPOCHS',
        'PPO_EPOCHS', 'PPO_MINIBATCH', 'MODEL_BATCH_SIZE', 'BATCH_SIZE', 'SEQ_LENGTH', 'rollout_max_length', 'HORIZON',
        'MPCHorizon', 'n_trajs', 'N_SAMPLES', 'use_MPCmodel']}
print(f"map={map_name} n_agents={n_agents} obs={cfg.IN_DIM} act={cfg.ACTION_SIZE} device={device} threads={torch.get_num_threads()}")
print("config:", desc)

learner = L.DreamerLearner(cfg)

def synth_episode(T):
    obs = np.random.randn(T, n_agents, cfg.IN_DIM).astype(np.float32)
    act = np.eye(cfg.ACTION_SIZE, dtype=np.float32)[np.random.randint(0, cfg.ACTION_SIZE, (T, n_agents))]
    rew = np.random.rand(T, n_agents, 1).astype(np.float32)
    done = np.zeros((T, n_agents, 1), np.float32); done[-1] = 1
    fake = np.zeros_like(done); last = np.zeros_like(done); last[-1] = 1
    av = np.ones((T, n_agents, cfg.ACTION_SIZE), np.float32)
    ep = {'observation': obs, 'action': act, 'reward': rew, 'done': done, 'fake': fake, 'last': last, 'avail_action': av}
    if hasattr(cfg, 'STATE_DIM'):
        ep['state'] = np.random.randn(T, cfg.STATE_DIM).astype(np.float32)
    return ep

def buffer_append(e):
    """Call replay_buffer.append by parameter name so MAG (7 args) and MABL (8 args, global_states 2nd) both work."""
    import inspect
    rb = learner.replay_buffer
    names = list(inspect.signature(rb.append).parameters)
    src = {'obs': 'observation', 'global_states': 'state', 'action': 'action', 'reward': 'reward', 'done': 'done',
           'fake': 'fake', 'last': 'last', 'av_action': 'avail_action'}
    rb.append(*[e[src[n]] for n in names])

for _ in range(20):
    buffer_append(synth_episode(60))
learner.total_samples = 20 * 60
learner.accum_samples = getattr(cfg, 'N_SAMPLES', 1)   # so a single step() call trains regardless of N_SAMPLES

timers, counts, peaks = {}, {}, {}
cuda = device == 'cuda'
def wrap(obj, name, key):
    f = getattr(obj, name, None)
    if f is None:
        return
    def g(*a, **k):
        if cuda: torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter(); r = f(*a, **k)
        if cuda: torch.cuda.synchronize()
        timers[key] = timers.get(key, 0.) + time.perf_counter() - t; counts[key] = counts.get(key, 0) + 1
        if cuda: peaks[key] = max(peaks.get(key, 0), torch.cuda.max_memory_allocated())
        return r
    setattr(obj, name, g)

wrap(learner, 'train_model', '1 model update (fwd+bwd, per epoch)')
wrap(L, 'get_model_loss_for_m_r_training', '2a m_r: no-grad model fwd')
wrap(learner, 'train_m_r_predictor', '2b m_r: predictor update')
wrap(L, 'actor_rollout', '3 actor_rollout (total)')
wrap(lossmod, 'rollout_representation', '3a   repr. rollout on real seq')
wrap(lossmod, 'rollout_policy', '3b   imagination rollout_policy')
wrap(rnns, 'MPCPredict', '3b-i   MPCPredict calls')
wrap(lossmod, 'critic_rollout', '3c   critic_rollout')
wrap(L, 'actor_loss', '4a PPO actor_loss fwd')
wrap(L, 'value_loss', '4b PPO value_loss fwd')
wrap(learner, 'apply_optimizer', '4c PPO apply_optimizer (bwd+step)')

roll = synth_episode(60)
if 'state' in roll:
    roll['global_state'] = roll['state']   # MABL's learner.step reads rollout['global_state']
t0 = time.perf_counter()
if cuda: torch.cuda.reset_peak_memory_stats()
learner.step(roll)
if cuda: torch.cuda.synchronize()
total = time.perf_counter() - t0
print(f"\nTOTAL learner.step = {total:.1f}s" + (f"   peak GPU mem (whole step) = {torch.cuda.max_memory_allocated()/1e9:.2f} GB" if cuda else "") + "\n")
hdr = f"{'section':46s} {'calls':>6s} {'total s':>9s} {'per call':>10s}" + (f" {'peak GB':>8s}" if cuda else "")
print(hdr)
for k in sorted(timers):
    line = f"{k:46s} {counts[k]:6d} {timers[k]:9.2f} {timers[k]/counts[k]:10.3f}"
    if cuda: line += f" {peaks.get(k, 0)/1e9:8.2f}"
    print(line)
