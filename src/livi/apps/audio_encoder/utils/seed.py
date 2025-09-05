import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
