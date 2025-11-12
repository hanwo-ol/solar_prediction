# 옵션 B(MC) 이분산 NLL + dynamics(속도/가속) 합성 데이터 항
import torch
import torch.nn.functional as F

def _get(cfg, key, default):
    return cfg[key] if (hasattr(cfg, "__contains__") and key in cfg) else default

@torch.no_grad()
def _ensure_timeweights(cfg, T: int):
    tw = _get(cfg, "TIME_WEIGHTS", None)
    if tw is None:
        return torch.ones(T, dtype=torch.float32)
    tw = torch.as_tensor(tw, dtype=torch.float32)
    if tw.numel() != T:
        # 길이가 다르면 균일 가중
        return torch.ones(T, dtype=torch.float32)
    return tw

def _mc_mean_std(fmodel, x, S: int, config=None):
    """
    수동 MC: fmodel(x, sample=True)를 S번 호출.
    표본 std(=diversity)가 작으면 입력 노이즈 주입으로 최대 3회까지 재시도.
    """
    def _stack_preds(x_):
        preds = []
        for _ in range(S):
            out = fmodel(x_, sample=True)                            # [B,C,H,W]
            out = torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)
            preds.append(out)
        preds = torch.stack(preds, dim=0)                            # [S,B,C,H,W]
        mean_pred = preds.mean(dim=0)
        std_pred  = preds.std(dim=0)
        return mean_pred, std_pred

    # 기본 시도
    mean_pred, std_pred = _stack_preds(x)

    # 다양성 체크/재시도
    thr = float(config.get("MC_DIVERSITY_THR", 1e-4)) if config else 1e-4
    base_noise = float(config.get("MC_INPUT_NOISE_STD", 5e-3)) if config else 5e-3
    max_tries = int(config.get("MC_DIVERSITY_MAX_TRIES", 3)) if config else 3

    tries, noise = 0, base_noise
    while std_pred.mean().item() < thr and tries < max_tries:
        noise = noise * 2.0 if tries > 0 else noise      # 5e-3 → 1e-2 → 2e-2 …
        x_noisy = x + torch.randn_like(x) * noise
        mean_pred, std_pred = _stack_preds(x_noisy)
        tries += 1

    # 마지막 안전화
    mean_pred = torch.nan_to_num(mean_pred, nan=0.0, posinf=1e3, neginf=-1e3)
    std_pred  = torch.nan_to_num(std_pred,  nan=0.0, posinf=1e3, neginf=0.0)
    return mean_pred, std_pred

def _heteroscedastic_nll_from_mu_var(mu, var, y, time_weights=None, eps=1e-12):
    """
    mu,var,y: [B,C,H,W], C=T
    NLL = 시간가중 평균_{c} 평균_{b,h,w} [ (y-mu)^2/var + log var ]
    """
    mu  = torch.nan_to_num(mu, nan=0.0, posinf=1e3, neginf=-1e3)
    y   = torch.nan_to_num(y,  nan=0.0, posinf=1e3, neginf=-1e-3)
    var = torch.nan_to_num(var, nan=1e-6, posinf=1e2, neginf=1e-6)
    var = torch.clamp(var, min=max(eps, 1e-6), max=1e2)

    nll_per = ((y - mu)**2 / var) + torch.log(var)      # [B,C,H,W]
    per_c = nll_per.mean(dim=(0, 2, 3))                 # [C]

    if time_weights is not None:
        tw = time_weights.to(per_c.device, per_c.dtype) # [C]
        nll = (per_c * tw).sum() / (tw.sum() + 1e-12)
    else:
        nll = per_c.mean()
    return nll

def _velocity_loss(mu, y, reduction="mean"):
    """속도(1계 차분) L1 정합"""
    mu = torch.nan_to_num(mu, nan=0.0, posinf=1e3, neginf=-1e3)
    y  = torch.nan_to_num(y,  nan=0.0, posinf=1e3, neginf=-1e3)
    if mu.shape[1] < 2:
        return mu.new_tensor(0.0)
    dmu = mu[:, 1:] - mu[:, :-1]        # [B,T-1,H,W]
    dy  =  y[:, 1:] -  y[:, :-1]
    l = (dmu - dy).abs()
    return l.mean() if reduction == "mean" else l.sum()

def _accel_loss(mu, y, reduction="mean"):
    """가속도(2계 차분) L1 정합: x_t - 2*x_{t-1} + x_{t-2}"""
    mu = torch.nan_to_num(mu, nan=0.0, posinf=1e3, neginf=-1e3)
    y  = torch.nan_to_num(y,  nan=0.0, posinf=1e3, neginf=-1e3)
    if mu.shape[1] < 3:
        return mu.new_tensor(0.0)
    d2mu = mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]   # [B,T-2,H,W]
    d2y  =  y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]
    l = (d2mu - d2y).abs()
    return l.mean() if reduction == "mean" else l.sum()

def data_term_mc_dynamics(fmodel, x, y, config):
    """
    옵션 B(MC) 이분산 NLL + dynamics(vel/acc)의 합성 데이터 항
    fmodel : (적응 중인) 베이지안 모델
    x, y   : [B, Cin, H, W], [B, C=T, H, W]
    config : dict (없으면 기본값 사용)
    반환   : data_loss (scalar), (mu, var)
    """
    S     = int(_get(config, "MC_INNER_SAMPLES", 4))
    tau2  = float(_get(config, "NLL_TAU2", 1e-4))
    w_vel = float(_get(config, "W_VEL", 0.20))
    w_acc = float(_get(config, "W_ACC", 0.05))

    T = y.shape[1]
    time_weights = _ensure_timeweights(config, T).to(y.device, y.dtype)

    # (1) MC 평균/분산
    mean_pred, std_pred = _mc_mean_std(fmodel, x, S, config=config)
    var = (std_pred**2) + tau2
    var = torch.nan_to_num(var, nan=1e-6, posinf=1e2, neginf=1e-6)
    var = torch.clamp(var, min=1e-6, max=1e2)

    # (2) 분산 인플레이션: diversity가 너무 작을 때만 소량 가산
    diversity = std_pred.mean().detach()
    infl_alpha = float(_get(config, "VAR_INFLATE_ALPHA", 0.05))  # 기본 0.05
    if diversity < float(_get(config, "MC_DIVERSITY_THR", 1e-4)):
        resid2 = (torch.nan_to_num(y - mean_pred))**2
        var = var + infl_alpha * resid2
        var = torch.clamp(var, min=1e-6, max=1e2)

    # (3) NLL + dynamics
    nll  = _heteroscedastic_nll_from_mu_var(mean_pred, var, y, time_weights=time_weights)
    with torch.no_grad():
        resid = (y - mean_pred)
        print(f"[dbg] mean|resid|: {resid.abs().mean().item():.4f}, "
              f"mean var: {var.mean().item():.6f}, "
              f"log var mean: {torch.log(var).mean().item():.4f}")
    lvel = _velocity_loss(mean_pred, y, reduction="mean")
    lacc = _accel_loss(mean_pred, y, reduction="mean")

    data_loss = nll + (w_vel * lvel) + (w_acc * lacc)
    return data_loss, (mean_pred, var)

# -----------------------------
# MetaLearner에 바로 꽂는 래퍼
# -----------------------------

def inner_total_loss_mc_dynamics(meta_learner, fmodel, support_x, support_y):
    """inner loop: data_term + beta * KL"""
    data_loss, _ = data_term_mc_dynamics(
        fmodel=fmodel,
        x=support_x,
        y=support_y,
        config=meta_learner.config
    )
    kl = fmodel.kl_divergence(meta_learner.prior_net)
    kl = torch.nan_to_num(kl, nan=0.0, posinf=1e6, neginf=1e6)  # KL 가드
    beta = float(_get(meta_learner.config, "KL_WEIGHT", 0.0))
    return data_loss + beta * kl

def outer_data_loss_mc_dynamics(meta_learner, fmodel, query_x, query_y):
    """outer loop: data_term (KL 없음)"""
    cfg = dict(meta_learner.config)
    cfg["MC_INNER_SAMPLES"] = int(_get(meta_learner.config, "MC_OUTER_SAMPLES",
                                _get(meta_learner.config, "MC_INNER_SAMPLES", 4)))
    data_loss, _ = data_term_mc_dynamics(
        fmodel=fmodel,
        x=query_x,
        y=query_y,
        config=cfg
    )
    return data_loss
