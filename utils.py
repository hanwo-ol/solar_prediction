# utils.py
import torch
import random
import os
import numpy as np

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def get_device():
    """Gets the target device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU: ", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    return device