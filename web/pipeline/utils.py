import gc
import torch
import os
import numpy as np

def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def seed_everything(seed):
    """
    Обеспечивает воспроизводимость экспериментов
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
