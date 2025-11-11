# loss_mc_dynamics.py
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

def _mc_mean_std(fmodel, x, S: int):
    """
    항상 수동 MC: fmodel(x, sample=True)를 S번 호출.
    모델이 num_samples 인자를 받지 않아도 호환.
    비정상값은 즉시 nan_to_num으로 치환.
    """
    preds = []
    for _ in range(S):
        out = fmodel(x, sample=True)                 # [B,C,H,W]
        out = torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)
        preds.append(out)
    preds = torch.stack(preds, dim=0)               # [S,B,C,H,W]
    mean_pred = preds.mean(dim=0)
    std_pred  = preds.std(dim=0)
    mean_pred = torch.nan_to_num(mean_pred, nan=0.0, posinf=1e3, neginf=-1e3)
    std_pred  = torch.nan_to_num(std_pred,  nan=0.0, posinf=1e3, neginf=0.0)
    return mean_pred, std_pred

def _mc_mu_var(fmodel, x, S: int, tau2: float):
    """
    MC 평균/표준편차 -> 분산 = std^2 + tau2.
    분산은 detach로 grad 차단(안정/메모리 절감).
    상/하한 및 nan/inf 치환으로 수치 안전화.
    """
    mean_pred, std_pred = _mc_mean_std(fmodel, x, S)
    var = (std_pred**2) + tau2
    var = torch.nan_to_num(var, nan=1e-6, posinf=1e2, neginf=1e-6)
    var = torch.clamp(var, min=1e-6, max=1e2)
    return mean_pred, var

def _heteroscedastic_nll_from_mu_var(mu, var, y, time_weights=None, eps=1e-12):
    """
    mu,var,y: [B,C,H,W], C=T
    NLL = 시간가중 평균_{c} 평균_{b,h,w} [ (y-mu)^2/var + log var ]
    """
    mu  = torch.nan_to_num(mu, nan=0.0, posinf=1e3, neginf=-1e3)
    y   = torch.nan_to_num(y,  nan=0.0, posinf=1e3, neginf=-1e3)
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
    """
    속도(1계 차분) L1 정합
    """
    mu = torch.nan_to_num(mu, nan=0.0, posinf=1e3, neginf=-1e3)
    y  = torch.nan_to_num(y,  nan=0.0, posinf=1e3, neginf=-1e3)
    if mu.shape[1] < 2:
        return mu.new_tensor(0.0)
    dmu = mu[:, 1:] - mu[:, :-1]        # [B,T-1,H,W]
    dy  =  y[:, 1:] -  y[:, :-1]
    l = (dmu - dy).abs()
    return l.mean() if reduction == "mean" else l.sum()

def _accel_loss(mu, y, reduction="mean"):
    """
    가속도(2계 차분) L1 정합: x_t - 2*x_{t-1} + x_{t-2}
    """
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
    S     = int(_get(config, "MC_INNER_SAMPLES", 4))  # inner 기본
    tau2  = float(_get(config, "NLL_TAU2", 1e-4))
    w_vel = float(_get(config, "W_VEL", 0.20))
    w_acc = float(_get(config, "W_ACC", 0.05))

    T = y.shape[1]
    time_weights = _ensure_timeweights(config, T).to(y.device, y.dtype)

    mu, var = _mc_mu_var(fmodel, x, S, tau2)

    nll  = _heteroscedastic_nll_from_mu_var(mu, var, y, time_weights=time_weights)
    with torch.no_grad():
        resid = (y - mu)
        print(f"[dbg] mean|resid|: {resid.abs().mean().item():.4f}, "
            f"mean var: {var.mean().item():.6f}, "
            f"log var mean: {torch.log(var).mean().item():.4f}")
    lvel = _velocity_loss(mu, y, reduction="mean")
    lacc = _accel_loss(mu, y, reduction="mean")

    data_loss = nll + (w_vel * lvel) + (w_acc * lacc)
    return data_loss, (mu, var)

# -----------------------------
# MetaLearner에 바로 꽂는 래퍼
# -----------------------------

def inner_total_loss_mc_dynamics(meta_learner, fmodel, support_x, support_y):
    """
    inner loop: data_term + beta * KL
    """
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
    """
    outer loop: data_term (KL 없음)
    """
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
