# meta_model_grad_b.py
# ------------------------------------------------------------
# Forward-only conditional heads with Residual-Delta form
# - y_{t+k} = y_{t+k-1}.detach() + Delta([h, y_{t+k-1}.detach()])
# - Bayesian U-Net backbone (GroupNorm)
# - KL over Bayesian layers only (paired with prior_net)
# - Compatible with loss_mc_dynamics.data_term_mc_dynamics
# ------------------------------------------------------------

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_mc_dynamics import data_term_mc_dynamics


# ============================================================
# Bayesian Base Layers
# ============================================================

class BayesianLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu: Optional[torch.Tensor] = None
        self.rho: Optional[torch.Tensor] = None

    def sample(self) -> torch.Tensor:
        sigma = F.softplus(self.rho)
        eps = torch.randn_like(sigma)
        return self.mu + sigma * eps

    def _kl_gaussian(self, mu_q, rho_q, mu_p, rho_p) -> torch.Tensor:
        # q ~ N(mu_q, softplus(rho_q)^2), p ~ N(mu_p, softplus(rho_p)^2)
        eps = 1e-8
        sigma_q = F.softplus(rho_q) + eps
        sigma_p = F.softplus(rho_p) + eps
        kl = (
            torch.log(sigma_p / sigma_q)
            + (sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * sigma_p.pow(2))
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
        nn.init.normal_(self.rho, mean=-3.0, std=0.1)
        nn.init.zeros_(self.bias_mu)
        nn.init.normal_(self.bias_rho, mean=-3.0, std=0.1)

    def forward(self, x, sample: bool = True):
        if sample:
            w = self.sample()
            b_sigma = F.softplus(self.bias_rho)
            b = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            w = self.mu
            b = self.bias_mu
        return F.conv2d(x, w, b, **self.kwargs)

    def kl_divergence(self, prior_layer: "BayesianConv2d") -> torch.Tensor:
        wkl = self._kl_gaussian(self.mu, self.rho, prior_layer.mu, prior_layer.rho)
        bkl = self._kl_gaussian(self.bias_mu, self.bias_rho,
                                prior_layer.bias_mu, prior_layer.bias_rho)
        return wkl + bkl


class BayesianConvTranspose2d(BayesianLayer):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.kwargs = kwargs
        # Weight shape for conv_transpose2d: (in_c, out_c, kH, kW)
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

    def forward(self, x, sample: bool = True):
        if sample:
            w = self.sample()
            b_sigma = F.softplus(self.bias_rho)
            b = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            w = self.mu
            b = self.bias_mu
        return F.conv_transpose2d(x, w, b, **self.kwargs)

    def kl_divergence(self, prior_layer: "BayesianConvTranspose2d") -> torch.Tensor:
        wkl = self._kl_gaussian(self.mu, self.rho, prior_layer.mu, prior_layer.rho)
        bkl = self._kl_gaussian(self.bias_mu, self.bias_rho,
                                prior_layer.bias_mu, prior_layer.bias_rho)
        return wkl + bkl


# ============================================================
# Blocks (GroupNorm 기본)
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
        self.conv1 = BayesianConv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.n1 = _gn(mid_channels, groups)
        self.conv2 = BayesianConv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.n2 = _gn(out_channels, groups)
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
            self.block = BayesianDoubleConv(in_channels, out_channels,
                                            mid_channels=in_channels // 2, groups=groups)
        else:
            self.up = BayesianConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.block = BayesianDoubleConv(in_channels, out_channels, groups=groups)

    def forward(self, x1, x2, sample: bool = True):
        if self.bilinear:
            x1 = self.up(x1)
        else:
            x1 = self.up(x1, sample)
        # spatial align
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.block(x, sample)


# ------------------------------------------------------------
# Residual-Delta Conditional Head (deterministic)
# ------------------------------------------------------------

class CondEmbed(nn.Module):
    """Embed previous frame (detached) with 3x3 conv + GN + ReLU."""
    def __init__(self, in_ch=1, emb_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, emb_ch, kernel_size=3, padding=1, bias=False),
            _gn(emb_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, y_prev_det: torch.Tensor) -> torch.Tensor:
        return self.net(y_prev_det)


class FuseBlock(nn.Module):
    """Fuse [h, prev_emb] with 3x3 conv + GN + ReLU."""
    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Head0(nn.Module):
    """First step head: non-linear 3x3 -> 1x1 (deterministic)"""
    def __init__(self, in_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            _gn(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class DeltaHead(nn.Module):
    """
    Residual-Delta head:
      y_{t+k} = y_{t+k-1}.detach() + Delta([h, Emb(y_{t+k-1}.detach())])
    """
    def __init__(self, h_ch=64, prev_ch=1, emb_ch=16):
        super().__init__()
        self.prev_embed = CondEmbed(prev_ch, emb_ch)
        self.fuse = FuseBlock(h_ch + emb_ch, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, h: torch.Tensor, y_prev_det: torch.Tensor) -> torch.Tensor:
        e = self.prev_embed(y_prev_det)
        z = self.fuse(torch.cat([h, e], dim=1))
        delta = self.out(z)
        return y_prev_det + delta


# ============================================================
# Bayesian U-Net with Residual-Delta Heads
# ============================================================

def _collect_bayesian_layers(model: nn.Module) -> List[nn.Module]:
    return [m for m in model.modules() if isinstance(m, (BayesianConv2d, BayesianConvTranspose2d))]


class BayesianUNet(nn.Module):
    """
    Backbone: Bayesian U-Net → shared feature h:[B,64,H,W]
    Heads:
      y_{t+1} = Head0(h)
      y_{t+k} = y_{t+k-1}.detach() + DeltaHead([h, y_{t+k-1}.detach()])
    """
    def __init__(self, n_channels: int, num_future_steps: int, bilinear: bool = True, groups: Optional[int] = None):
        super().__init__()
        self.num_future_steps = int(num_future_steps)

        # Encoder / Decoder
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
        self.head0 = Head0(64)
        self.head_next = nn.ModuleList([
            DeltaHead(h_ch=64, prev_ch=1, emb_ch=16) for _ in range(self.num_future_steps - 1)
        ])

    # decode shared features
    def _decode_features(self, x, sample: bool) -> torch.Tensor:
        x1 = self.inc(x, sample)
        x2 = self.down1(x1, sample)
        x3 = self.down2(x2, sample)
        x4 = self.down3(x3, sample)
        x5 = self.down4(x4, sample)
        x  = self.up1(x5, x4, sample)
        x  = self.up2(x,  x3, sample)
        x  = self.up3(x,  x2, sample)
        h  = self.up4(x,  x1, sample)  # [B,64,H,W]
        return h

    def forward(self, x, sample: bool = True):
        """
        x: [B, Cin, H, W]
        return: [B, T, H, W]
        """
        h = self._decode_features(x, sample)
        y0 = self.head0(h)                # [B,1,H,W]
        ys = [y0]
        for k in range(self.num_future_steps - 1):
            yk = self.head_next[k](h, ys[-1].detach())
            ys.append(yk)
        y = torch.cat(ys, dim=1)
        return y

    def kl_divergence(self, prior_model: "BayesianUNet") -> torch.Tensor:
        # Pair only Bayesian layers to avoid ordering issues
        kl_total = torch.tensor(0.0, device=next(self.parameters()).device)
        mine = _collect_bayesian_layers(self)
        prior = _collect_bayesian_layers(prior_model)
        assert len(mine) == len(prior), "Bayesian layer count mismatch between model and prior!"
        for m, p in zip(mine, prior):
            kl_total = kl_total + m.kl_divergence(p)
        return kl_total


# ============================================================
# Meta Learner (compatible interface)
# ============================================================

class MetaLearner(nn.Module):
    """
    - prior_net's Bayesian parameters are meta-learned
    - inner_loop_loss adds KL, outer_loop_loss is data term only
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.prior_net = BayesianUNet(config['INPUT_LEN'], config['TARGET_LEN'])

    def forward(self, x, sample: bool = True, num_samples: int = 1):
        # For convenience in evaluation: return mean/std when num_samples>1
        if (not sample) or (num_samples == 1):
            return self.prior_net(x, sample)
        preds = [self.prior_net(x, sample=True) for _ in range(num_samples)]  # [T] each
        preds = torch.stack(preds, dim=0)  # [S,B,T,H,W]
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)
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
