import random
import numpy as np
import torch


# Sets deterministic random seeds across PyTorch, NumPy, and Python for reproducible experiments.
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
