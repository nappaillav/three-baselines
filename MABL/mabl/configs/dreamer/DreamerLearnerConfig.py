from agent.learners.DreamerLearner import DreamerLearner
from configs.dreamer.DreamerAgentConfig import DreamerConfig


class DreamerLearnerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_LR = 3e-4
        self.ACTOR_LR = 1e-4
        self.VALUE_LR = 1e-3
        print(self.MODEL_LR, self.ACTOR_LR, self.VALUE_LR)
        self.CAPACITY = 250000
        self.MIN_BUFFER_SIZE = 500
        self.MODEL_EPOCHS = 20          # step D (was 60): 20 x batch 120 = same 2,400 sequences/cycle
        self.EPOCHS = 4                 # since step C: multiplier of BATCH_SIZE for the single batched imagination rollout
        self.PPO_EPOCHS = 5
        self.PPO_MINIBATCH = 1000       # step C: PPO minibatch rows (was the literal 2000 in train_agent)
        self.MODEL_BATCH_SIZE = 120     # step D (was 40)
        self.BATCH_SIZE = 40
        self.SEQ_LENGTH = 20
        self.N_SAMPLES = 500            # step D (was 1): transitions between learner cycles
        self.TARGET_UPDATE = 1
        self.DEVICE = 'cuda'
        self.GRAD_CLIP = 100.0
        self.HORIZON = 5                # step D (was 15): imagination length (MABL's rollout_policy uses HORIZON)
        self.ENTROPY = 0.001
        self.ENTROPY_ANNEALING = 0.99998
        self.GRAD_CLIP_POLICY = 100.0

    def create_learner(self):
        return DreamerLearner(self)
