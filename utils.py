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


# ------- Metrics & Logging helpers -------
import torch.nn.functional as F
from pathlib import Path as _Path
import csv, time, json as _json

def _gaussian_window(ch, win_size=11, sigma=1.5, device='cpu', dtype=torch.float32):
    import torch
    coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
    g = torch.exp(-(coords**2)/(2*sigma*sigma))
    g = (g / g.sum()).view(1,1,1,-1)
    w = g.transpose(-1,-2) @ g  # 2D
    w = w / w.sum()
    w = w.view(1,1,win_size,win_size).repeat(ch,1,1,1)
    return w

@torch.no_grad()
def ssim_tchw(x, y, data_range=1.0, win_size=11, K1=0.01, K2=0.03):
    """
    x,y: [B,T,H,W] float
    returns: scalar mean SSIM
    """
    import torch
    assert x.shape == y.shape and x.dim()==4
    B,T,H,W = x.shape
    C = T  # treat T as channels
    x = x.view(B,C,H,W)
    y = y.view(B,C,H,W)
    device, dtype = x.device, x.dtype
    window = _gaussian_window(C, win_size=win_size, device=device, dtype=dtype)

    C1 = (K1*data_range)**2
    C2 = (K2*data_range)**2

    mu_x = F.conv2d(x, window, groups=C, padding=win_size//2)
    mu_y = F.conv2d(y, window, groups=C, padding=win_size//2)
    mu_x2 = mu_x*mu_x
    mu_y2 = mu_y*mu_y
    mu_xy = mu_x*mu_y

    sigma_x2 = F.conv2d(x*x, window, groups=C, padding=win_size//2) - mu_x2
    sigma_y2 = F.conv2d(y*y, window, groups=C, padding=win_size//2) - mu_y2
    sigma_xy = F.conv2d(x*y, window, groups=C, padding=win_size//2) - mu_xy

    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1)*(sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean().item()

@torch.no_grad()
def compute_metrics(y_pred_mean, y_true, data_range=1.0):
    import torch
    y_pred_mean = torch.nan_to_num(y_pred_mean, nan=0.0, posinf=1e3, neginf=-1e3)
    y_true      = torch.nan_to_num(y_true,      nan=0.0, posinf=1e3, neginf=-1e3)
    mse = F.mse_loss(y_pred_mean, y_true).item()
    mae = F.l1_loss(y_pred_mean, y_true).item()
    ssim = ssim_tchw(y_pred_mean, y_true, data_range=float(data_range))
    return {"mse": mse, "mae": mae, "ssim": ssim}

def start_run_log(base_dir="log", config_dict=None):
    ts = __import__('time').strftime("%Y%m%d-%H%M%S")
    run_dir = _Path(base_dir)/ts
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_dict is not None:
        (run_dir/"config.json").write_text(__import__('json').dumps(config_dict, indent=2), encoding="utf-8")
    return str(run_dir)

class RunLogger:
    def __init__(self, run_dir: str):
        self.run_dir = _Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        if not self.csv_path.exists():
            self.csv_path.write_text("epoch,split,loss,mse,mae,ssim,kl_weight,notes\n", encoding="utf-8")

    def log_row(self, epoch, split, loss, metrics: dict, kl_weight=None, notes=""):
        with self.csv_path.open("a", newline="") as f:
            f.write(f"{epoch},{split},{'' if loss is None else f'{loss:.6f}'},{metrics.get('mse','')},{metrics.get('mae','')},{metrics.get('ssim','')},{'' if kl_weight is None else f'{kl_weight:.6e}'},{notes}\\n")
