# no-op stand-in so the fork's `import wandb` resolves on MARLenv; never called because use_wandb=False in the profiles
def init(*a, **k): pass
def log(*a, **k): pass
def define_metric(*a, **k): pass
