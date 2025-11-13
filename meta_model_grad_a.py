# meta_model_grad_a.py
# ------------------------------------------------------------
# Forward-only conditional heads (with nonlinear/spatial context)
# - U-Net body: Bayesian conv blocks (MC sampling & KL)
# - Heads: deterministic convs with 3x3 + GN + ReLU before 1x1
# - t+1 uses h only; t+k (k>=2) uses [h, y_{t+k-1}.detach()] as condition
# - Loss: use loss_mc_dynamics.data_term_mc_dynamics (Option B MC + dynamics)
# - KL: only over Bayesian layers (body), prior_net has identical structure
# ------------------------------------------------------------

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_mc_dynamics import data_term_mc_dynamics


# ============================================================
# Bayesian base layers
# ============================================================

class BayesianLayer(nn.Module):
    """Base for Bayesian layers with (mu, rho) parameterization."""
    def __init__(self):
        super().__init__()
        self.mu: Optional[torch.Tensor] = None
        self.rho: Optional[torch.Tensor] = None

    def _sample_weight(self) -> torch.Tensor:
        sigma = F.softplus(self.rho)
        eps = torch.randn_like(sigma)
        return self.mu + sigma * eps

    @staticmethod
    def _kl_gaussian(mu_q, sigma_q, mu_p, sigma_p) -> torch.Tensor:
        # KL[N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)] summed over all params
        eps = 1e-8
        sigma_q = sigma_q + eps
        sigma_p = sigma_p + eps
        kl = (
            torch.log(sigma_p / sigma_q)
            + (sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / (2.0 * sigma_p.pow(2))
            - 0.5
        )
        return kl.sum()

    def kl_divergence(self, prior_layer: "BayesianLayer") -> torch.Tensor:
        raise NotImplementedError


class BayesianConv2d(BayesianLayer):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        # Guard: bias kwarg must be ignored because this layer manages bias internally.
        if 'bias' in kwargs:
            kwargs.pop('bias')
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.kwargs = kwargs
        self.mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *k))
        self.rho = nn.Parameter(torch.Tensor(out_channels, in_channels, *k))
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mu, a=0.01)
        nn.init.normal_(self.rho, mean=-3.0, std=0.1)
        nn.init.zeros_(self.bias_mu)
        nn.init.normal_(self.bias_rho, mean=-3.0, std=0.1)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            w = self._sample_weight()
            b_sigma = F.softplus(self.bias_rho)
            b = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            w = self.mu
            b = self.bias_mu
        return F.conv2d(x, w, b, **self.kwargs)

    def kl_divergence(self, prior_layer: "BayesianConv2d") -> torch.Tensor:
        wkl = self._kl_gaussian(
            self.mu, F.softplus(self.rho),
            prior_layer.mu, F.softplus(prior_layer.rho)
        )
        bkl = self._kl_gaussian(
            self.bias_mu, F.softplus(self.bias_rho),
            prior_layer.bias_mu, F.softplus(prior_layer.bias_rho)
        )
        return wkl + bkl


class BayesianConvTranspose2d(BayesianLayer):
    # Weight shape for conv_transpose2d is (in_channels, out_channels, k, k)
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        # Guard: bias kwarg must be ignored because this layer manages bias internally.
        if 'bias' in kwargs:
            kwargs.pop('bias')
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.kwargs = kwargs
        self.mu = nn.Parameter(torch.Tensor(in_channels, out_channels, *k))
        self.rho = nn.Parameter(torch.Tensor(in_channels, out_channels, *k))
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mu, a=0.01)
        nn.init.normal_(self.rho, mean=-3.0, std=0.1)
        nn.init.zeros_(self.bias_mu)
        nn.init.normal_(self.bias_rho, mean=-3.0, std=0.1)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            w = self._sample_weight()
            b_sigma = F.softplus(self.bias_rho)
            b = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            w = self.mu
            b = self.bias_mu
        return F.conv_transpose2d(x, w, b, **self.kwargs)

    def kl_divergence(self, prior_layer: "BayesianConvTranspose2d") -> torch.Tensor:
        wkl = self._kl_gaussian(
            self.mu, F.softplus(self.rho),
            prior_layer.mu, F.softplus(prior_layer.rho)
        )
        bkl = self._kl_gaussian(
            self.bias_mu, F.softplus(self.bias_rho),
            prior_layer.bias_mu, F.softplus(prior_layer.bias_rho)
        )
        return wkl + bkl


# ============================================================
# Blocks (GroupNorm-based)
# ============================================================

def _gn(c: int, groups: Optional[int] = None) -> nn.GroupNorm:
    if groups is None:
        g = max(1, c // 8)
        g = min(32, g)
    else:
        g = max(1, min(groups, c))
    return nn.GroupNorm(g, c)

class BayesianDoubleConv(nn.Module):
    """(Bayesian conv -> GN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, groups: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = BayesianConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True)  # bias kwarg ignored inside
        self.n1 = _gn(mid_channels, groups)
        self.conv2 = BayesianConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)  # bias kwarg ignored
        self.n2 = _gn(out_channels, groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        return self.block(self.pool(x), sample)

class BayesianUp(nn.Module):
    """Upsample (bilinear or Bayesian deconv) + double conv."""
    def __init__(self, in_channels, out_channels, bilinear=True, groups: Optional[int] = None):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.block = BayesianDoubleConv(in_channels, out_channels, mid_channels=in_channels // 2, groups=groups)
        else:
            self.up = BayesianConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=True)  # bias kwarg ignored
            self.block = BayesianDoubleConv(in_channels, out_channels, groups=groups)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if self.bilinear:
            x1 = self.up(x1)
        else:
            x1 = self.up(x1, sample)
        # pad to match skip
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.block(x, sample)


# ============================================================
# Conditional heads (non-Bayesian, with spatial context)
# ============================================================

class Head0(nn.Module):
    """First time-step head: (3x3 + GN + ReLU) -> 1x1"""
    def __init__(self, in_ch=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            _gn(64, groups=8),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.f(h)
        return self.out(z)


class CondHead(nn.Module):
    """Next time-step head: prev 3x3-embed + concat with h -> 3x3 block -> 1x1"""
    def __init__(self, h_ch=64, prev_ch=1, emb_ch=16):
        super().__init__()
        self.prev_embed = nn.Sequential(
            nn.Conv2d(prev_ch, emb_ch, kernel_size=3, padding=1, bias=False),
            _gn(emb_ch),
            nn.ReLU(inplace=True),
        )
        in_ch = h_ch + emb_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            _gn(64, groups=8),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, h: torch.Tensor, y_prev_det: torch.Tensor) -> torch.Tensor:
        e = self.prev_embed(y_prev_det)
        z = self.fuse(torch.cat([h, e], dim=1))
        return self.out(z)


# ============================================================
# Bayesian U-Net with conditional (detach) heads
# ============================================================

def _collect_bayesian_layers(model: nn.Module) -> List[nn.Module]:
    return [m for m in model.modules() if isinstance(m, BayesianLayer)]

class BayesianUNet(nn.Module):
    """
    Body: Bayesian U-Net -> shared feature h:[B,64,H,W]
    Heads:
      y_{t+1} = Head0(h)
      y_{t+k} = CondHead([h, y_{t+k-1}.detach()])  (k>=2)
    Output: [B, T, H, W]
    """
    def __init__(self, n_channels: int, num_future_steps: int, bilinear: bool = True, groups: Optional[int] = None):
        super().__init__()
        self.num_future_steps = int(num_future_steps)

        # Encoder/Decoder (Bayesian)
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

        # Conditional heads (deterministic)
        self.head0 = Head0(in_ch=64)
        if self.num_future_steps > 1:
            self.head_next = nn.ModuleList([CondHead(h_ch=64, prev_ch=1, emb_ch=16)
                                            for _ in range(self.num_future_steps - 1)])
        else:
            self.head_next = nn.ModuleList([])

    # shared feature
    def _decode_features(self, x: torch.Tensor, sample: bool) -> torch.Tensor:
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

    def _single_pass(self, x: torch.Tensor, sample: bool) -> torch.Tensor:
        h = self._decode_features(x, sample)        # [B,64,H,W]
        y0 = self.head0(h)                          # [B,1,H,W]
        ys = [y0]
        for k in range(self.num_future_steps - 1):
            cond_prev = ys[-1].detach()
            yk = self.head_next[k](h, cond_prev)
            ys.append(yk)
        y = torch.cat(ys, dim=1)                   # [B,T,H,W]
        return y

    def forward(self, x: torch.Tensor, sample: bool = True, num_samples: int = 1):
        """
        If num_samples==1 or sample==False -> return [B,T,H,W]
        Else (MC) -> return (mean:[B,T,H,W], std:[B,T,H,W])
        """
        if (not sample) or (num_samples == 1):
            return self._single_pass(x, sample)

        # MC posterior predictive
        preds = []
        for _ in range(num_samples):
            preds.append(self._single_pass(x, sample=True))
        preds = torch.stack(preds, dim=0)               # [S,B,T,H,W]
        mean = preds.mean(dim=0)
        std  = preds.std(dim=0, unbiased=True)
        return mean, std

    def kl_divergence(self, prior_model: "BayesianUNet") -> torch.Tensor:
        """Sum KL over Bayesian layers only (heads are deterministic)."""
        kl_total = 0.0
        mine = _collect_bayesian_layers(self)
        prior = _collect_bayesian_layers(prior_model)
        assert len(mine) == len(prior), "Bayesian layer count mismatch between model and prior!"
        for m, p in zip(mine, prior):
            kl_total = kl_total + m.kl_divergence(p)
        return kl_total


# ============================================================
# Meta-learner wrapper (compatible with engine/loss)
# ============================================================

class MetaLearner(nn.Module):
    """
    - Holds the meta-parameters as prior_net (Bayesian U-Net with conditional heads)
    - inner_loop_loss: data term (MC + dynamics) + beta * KL
    - outer_loop_loss: data term only
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.prior_net = BayesianUNet(
            n_channels=self.config['INPUT_LEN'],
            num_future_steps=self.config['TARGET_LEN'],
            bilinear=True,
            groups=None,
        )

    # Convenience interface used by some utilities (not engine-critical)
    def forward(self, x: torch.Tensor, sample: bool = True, num_samples: int = 1):
        return self.prior_net(x, sample=sample, num_samples=num_samples)

    def inner_loop_loss(self, fmodel: BayesianUNet, support_x: torch.Tensor, support_y: torch.Tensor):
        # data term (uses MC_INNER_SAMPLES in config)
        data_loss, _ = data_term_mc_dynamics(fmodel=fmodel, x=support_x, y=support_y, config=self.config)

        # KL between adapted fmodel and prior_net (Bayesian layers only)
        kl = fmodel.kl_divergence(self.prior_net)
        kl = torch.nan_to_num(kl, nan=0.0, posinf=1e6, neginf=1e6)

        beta = float(self.config.get("KL_WEIGHT", 0.0))
        total = data_loss + beta * kl
        total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=1e6)

        if torch.isnan(total):
            raise RuntimeError(f"[inner_loop_loss] NaN detected: data={float(data_loss.detach().cpu())}, kl={float(kl.detach().cpu())}, beta={beta}")
        return total

    def outer_loop_loss(self, fmodel: BayesianUNet, query_x: torch.Tensor, query_y: torch.Tensor):
        # Make sure outer uses MC_OUTER_SAMPLES if provided
        cfg = dict(self.config)
        if "MC_OUTER_SAMPLES" in cfg:
            cfg["MC_INNER_SAMPLES"] = int(cfg["MC_OUTER_SAMPLES"])
        data_loss, _ = data_term_mc_dynamics(fmodel=fmodel, x=query_x, y=query_y, config=cfg)
        data_loss = torch.nan_to_num(data_loss, nan=0.0, posinf=1e6, neginf=1e6)
        if torch.isnan(data_loss):
            raise RuntimeError(f"[outer_loop_loss] NaN detected: data={float(data_loss.detach().cpu())}")
        return data_loss
