import logging
import os
import random

import numpy as np
import torch


logger = logging.getLogger(__name__)

def seed(value=42, deterministic=True):
    logger.debug(f"Setting seed to {value}, deterministic={deterministic}")
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
