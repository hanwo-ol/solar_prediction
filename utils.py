# utils.py
import torch
import random
import os
import numpy as np

def set_seed(seed):
    """재현성을 위해 시드를 설정합니다."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # AMP 사용 시 benchmark=True가 성능에 더 유리할 수 있으나,
        # 재현성을 위해 False로 설정합니다.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def get_device():
    """사용할 장치를 가져옵니다."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    return device