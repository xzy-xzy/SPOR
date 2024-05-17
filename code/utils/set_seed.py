import torch as tch
import numpy as np
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tch.manual_seed(seed)
    tch.cuda.manual_seed_all(seed)
    tch.backends.cudnn.deterministic = True
