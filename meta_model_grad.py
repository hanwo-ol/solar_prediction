# meta_model_grad.py
# ------------------------------------------------------------
# Forward-only conditional heads (stop-grad) 을 적용한 파일럿 모델
# - Bayesian U-Net 본체는 유지
# - 출력 헤드를 시간별로 분리하고, t+1.. 시점은 [h, y_prev.detach()]를 조건으로 사용
# - KL 짝맞춤은 Bayesian 레이어만 추출하여 계산 (순서 오염 방지)
# - 기본 정규화: GroupNorm (작은 배치/메타러닝 안정화)
# - loss_mc_dynamics의 data_term_mc_dynamics와 완전 호환
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from loss_mc_dynamics import data_term_mc_dynamics

# ============================================================
# Bayesian Base
# ============================================================

class BayesianLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu: Optional[torch.Tensor] = None
        self.rho: Optional[torch.Tensor] = None

    def sample(self):
        sigma = F.softplus(self.rho)
        eps = torch.randn_like(sigma)
        return self.mu + sigma * eps

    def kl_divergence(self, prior_mu: torch.Tensor, prior_sigma: torch.Tensor):
        post_sigma = F.softplus(self.rho)
        eps = 1e-8
        prior_sigma = prior_sigma + eps
        post_sigma = post_sigma + eps
        kl = (
            torch.log(prior_sigma / post_sigma)
            + (post_sigma.pow(2) + (self.mu - prior_mu).pow(2)) / (2 * prior_sigma.pow(2))
            - 0.5
        )
        return kl.sum()


class BayesianConv2d(BayesianLayer):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.kwargs = kwargs
        self.mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *k))
        self.rho = nn.Parameter(torch.Tensor(out_channels, in_channels, *k))
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mu, a=0.01)
        nn.init.normal_(self.rho, mean=-3, std=0.1)
        nn.init.zeros_(self.bias_mu)
        nn.init.normal_(self.bias_rho, mean=-3, std=0.1)

    def forward(self, x, sample: bool = True):
        if sample:
            w = self.sample()
            b_sigma = F.softplus(self.bias_rho)
            b = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            w = self.mu
            b = self.bias_mu
        return F.conv2d(x, w, b, **self.kwargs)

    def kl_divergence(self, prior_layer: "BayesianConv2d"):
        wkl = super().kl_divergence(prior_layer.mu, F.softplus(prior_layer.rho))
        b_post = F.softplus(self.bias_rho)
        b_prior = F.softplus(prior_layer.bias_rho)
        bkl = (
            torch.log(b_prior / b_post)
            + (b_post.pow(2) + (self.bias_mu - prior_layer.bias_mu).pow(2)) / (2 * b_prior.pow(2))
            - 0.5
        )
        return wkl + bkl.sum()


class BayesianConvTranspose2d(BayesianLayer):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.kwargs = kwargs
        # conv_transpose2d의 weight shape는 (in_c, out_c, k, k)
        self.mu = nn.Parameter(torch.Tensor(in_channels, out_channels, *k))
        self.rho = nn.Parameter(torch.Tensor(in_channels, out_channels, *k))
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mu, a=0.01)
        nn.init.normal_(self.rho, mean=-3, std=0.1)
        nn.init.zeros_(self.bias_mu)
        nn.init.normal_(self.bias_rho, mean=-3, std=0.1)

    def forward(self, x, sample: bool = True):
        if sample:
            w = self.sample()
            b_sigma = F.softplus(self.bias_rho)
            b = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            w = self.mu
            b = self.bias_mu
        return F.conv_transpose2d(x, w, b, **self.kwargs)

    def kl_divergence(self, prior_layer: "BayesianConvTranspose2d"):
        wkl = super().kl_divergence(prior_layer.mu, F.softplus(prior_layer.rho))
        b_post = F.softplus(self.bias_rho)
        b_prior = F.softplus(prior_layer.bias_rho)
        bkl = (
            torch.log(b_prior / b_post)
            + (b_post.pow(2) + (self.bias_mu - prior_layer.bias_mu).pow(2)) / (2 * b_prior.pow(2))
            - 0.5
        )
        return wkl + bkl.sum()

# ============================================================
# Blocks (GroupNorm 기본)
# ============================================================

def _group_norm(c: int, groups: Optional[int] = None) -> nn.GroupNorm:
    if groups is None:
        g = max(1, c // 8)
        g = min(32, g)
    else:
        g = max(1, min(groups, c))
    return nn.GroupNorm(g, c)

class BayesianDoubleConv(nn.Module):
    """(Bayesian conv => GN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, groups: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = BayesianConv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.n1 = _group_norm(mid_channels, groups)
        self.conv2 = BayesianConv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.n2 = _group_norm(out_channels, groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, sample: bool = True):
        x = self.conv1(x, sample)
        x = self.n1(x)
        x = self.relu(x)
        x = self.conv2(x, sample)
        x = self.n2(x)
        x = self.relu(x)
        return x

class BayesianDown(nn.Module):
    def __init__(self, in_channels, out_channels, groups: Optional[int] = None):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = BayesianDoubleConv(in_channels, out_channels, groups=groups)

    def forward(self, x, sample: bool = True):
        return self.block(self.pool(x), sample)

class BayesianUp(nn.Module):
    """Upsample (bilinear or deconv) + double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, groups: Optional[int] = None):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.block = BayesianDoubleConv(in_channels, out_channels, mid_channels=in_channels // 2, groups=groups)
        else:
            self.up = BayesianConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.block = BayesianDoubleConv(in_channels, out_channels, groups=groups)

    def forward(self, x1, x2, sample: bool = True):
        if self.bilinear:
            x1 = self.up(x1)
        else:
            x1 = self.up(x1, sample)
        # pad to match
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.block(x, sample)

class BayesianOutConv(BayesianConv2d):
    """1x1 Bayesian head"""
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1)

# ============================================================
# Bayesian U-Net with forward-only conditional heads
# ============================================================

def _collect_bayesian_layers(model: nn.Module) -> List[nn.Module]:
    return [m for m in model.modules() if isinstance(m, BayesianLayer)]

class BayesianUNet(nn.Module):
    """
    - 디코더 출력 특징맵 h:[B,64,H,W]
    - y_0 = head0(h)
    - y_{k} = head_next[k-1]([h, y_{k-1}.detach()])
    -> 순전파 조건만 연결되고, 역전파는 y_{k-1} 경로로 흐르지 않음
    """
    def __init__(self, n_channels: int, num_future_steps: int, bilinear: bool = True, groups: Optional[int] = None):
        super().__init__()
        self.num_future_steps = int(num_future_steps)

        # Encoder/Decoder
        self.inc = BayesianDoubleConv(n_channels, 64, groups=groups)
        self.down1 = BayesianDown(64, 128, groups=groups)
        self.down2 = BayesianDown(128, 256, groups=groups)
        self.down3 = BayesianDown(256, 512, groups=groups)
        factor = 2 if bilinear else 1
        self.down4 = BayesianDown(512, 1024 // factor, groups=groups)
        self.up1 = BayesianUp(1024, 512 // factor, bilinear, groups=groups)
        self.up2 = BayesianUp(512, 256 // factor, bilinear, groups=groups)
        self.up3 = BayesianUp(256, 128 // factor, bilinear, groups=groups)
        self.up4 = BayesianUp(128, 64, bilinear, groups=groups)

        # Heads
        self.head0 = BayesianOutConv(64, 1)
        self.head_next = nn.ModuleList([
            BayesianOutConv(64 + 1, 1) for _ in range(self.num_future_steps - 1)
        ])

    # 내장: 디코더 끝 특징 h 생성
    def _decode_features(self, x, sample: bool):
        x1 = self.inc(x, sample)
        x2 = self.down1(x1, sample)
        x3 = self.down2(x2, sample)
        x4 = self.down3(x3, sample)
        x5 = self.down4(x4, sample)
        x  = self.up1(x5, x4, sample)
        x  = self.up2(x,  x3, sample)
        x  = self.up3(x,  x2, sample)
        h  = self.up4(x,  x1, sample)   # [B,64,H,W]
        return h

    def forward(self, x, sample: bool = True):
        """
        입력: x [B, Cin, H, W]
        출력: y [B, T, H, W]
        """
        h = self._decode_features(x, sample)        # [B,64,H,W]
        y0 = self.head0(h, sample)                  # [B,1,H,W]
        ys = [y0]
        for k in range(self.num_future_steps - 1):
            cond = torch.cat([h, ys[-1].detach()], dim=1)  # stop-grad
            yk = self.head_next[k](cond, sample)
            ys.append(yk)
        y = torch.cat(ys, dim=1)
        return y

    def kl_divergence(self, prior_model: "BayesianUNet") -> torch.Tensor:
        """Bayesian 레이어만 짝맞춰 KL 합산 (순서 오염 방지)"""
        kl_total = 0.0
        mine = _collect_bayesian_layers(self)
        prior = _collect_bayesian_layers(prior_model)
        assert len(mine) == len(prior), "Bayesian layer count mismatch between model and prior!"
        for m, p in zip(mine, prior):
            kl_total += m.kl_divergence(p)
        return kl_total

# ============================================================
# Meta Learner (loss와 인터페이스 호환)
# ============================================================

class MetaLearner(nn.Module):
    """
    - prior_net 파라미터(사전분포 파라미터)를 메타 업데이트
    - inner_loop_loss / outer_loop_loss는 기존 loss_mc_dynamics와 동일 인터페이스
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.prior_net = BayesianUNet(config['INPUT_LEN'], config['TARGET_LEN'])

    def forward(self, x, sample: bool = True, num_samples: int = 1):
        if (not sample) or (num_samples == 1):
            return self.prior_net(x, sample)
        preds = [self.prior_net(x, sample=True) for _ in range(num_samples)]
        preds = torch.stack(preds, dim=0)  # [S,B,T,H,W]
        mean_pred = preds.mean(dim=0)
        std_pred  = preds.std(dim=0)
        return mean_pred, std_pred

    def inner_loop_loss(self, fmodel: "BayesianUNet", support_x, support_y):
        data_loss, _ = data_term_mc_dynamics(fmodel=fmodel, x=support_x, y=support_y, config=self.config)
        kl = fmodel.kl_divergence(self.prior_net)
        kl = torch.nan_to_num(kl, nan=0.0, posinf=1e6, neginf=1e6)
        beta = float(self.config.get("KL_WEIGHT", 0.0))
        total = data_loss + beta * kl
        total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=1e6)
        if torch.isnan(total):
            raise RuntimeError(f"[inner_loop_loss] NaN detected: data={float(data_loss.detach().cpu())}, kl={float(kl.detach().cpu())}, beta={beta}")
        return total

    def outer_loop_loss(self, fmodel: "BayesianUNet", query_x, query_y):
        cfg = dict(self.config)
        if "MC_OUTER_SAMPLES" in cfg:
            cfg["MC_INNER_SAMPLES"] = int(cfg["MC_OUTER_SAMPLES"])
        data_loss, _ = data_term_mc_dynamics(fmodel=fmodel, x=query_x, y=query_y, config=cfg)
        if torch.isnan(data_loss):
            raise RuntimeError(f"[outer_loop_loss] NaN detected: data={float(data_loss.detach().cpu())}")
        return data_loss
