# loss_mc_dynamics.py
# MC 이분산 NLL + dynamics(속도/가속) 합성 데이터 항
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
        # 길이가 다르면 균일 가중으로 대체
        return torch.ones(T, dtype=torch.float32)
    return tw

def _mc_mean_std(fmodel, x, S: int):
    """
    fmodel(x, sample=True)를 S번 호출해 [S,B,C,H,W]를 만들고
    평균/표준편차 반환. num_samples 인자는 사용하지 않음.
    """
    preds = []
    for _ in range(S):
        preds.append(fmodel(x, sample=True))  # BayesianUNet.forward(x, sample)
    preds = torch.stack(preds, dim=0)  # [S,B,C,H,W]
    return preds.mean(dim=0), preds.std(dim=0)

def _heteroscedastic_nll_from_mu_var(mu, var, y, time_weights=None, eps=1e-12):
    """
    mu,var,y: [B,C,H,W], time_weights: [C] or None
    NLL = mean_{b,c,h,w} [ (y-mu)^2/var + log var ]
    시간 가중이 있으면 channel(C) 축에 곱해 적용
    """
    # 안정화
    var = torch.clamp(var, min=eps)
    nll_per = ( (y - mu)**2 / var ) + torch.log(var)  # [B,C,H,W]

    if time_weights is not None:
        # time_weights: [C] → [1,C,1,1]
        tw = time_weights.view(1, -1, 1, 1).to(nll_per.device, nll_per.dtype)
        nll_per = nll_per * tw
        denom = nll_per.new_tensor(tw.sum().item())
    else:
        denom = nll_per.new_tensor(nll_per.shape[1])  # C

    # [B,C,H,W] → 평균
    nll = nll_per.mean(dim=(0,2,3)).sum() / denom  # 채널가중 정규화
    return nll

def _velocity_loss(mu, y, reduction="mean"):
    """
    속도(1계 차분) L1 정합
    mu,y: [B,C,H,W], C=T
    """
    if mu.shape[1] < 2:
        return mu.new_tensor(0.0)
    dmu = mu[:,1:] - mu[:,:-1]        # [B,T-1,H,W]
    dy  =  y[:,1:] -  y[:,:-1]
    l = (dmu - dy).abs()
    return l.mean() if reduction=="mean" else l.sum()

def _accel_loss(mu, y, reduction="mean"):
    """
    가속도(2계 차분) L1 정합
    """
    if mu.shape[1] < 3:
        return mu.new_tensor(0.0)
    d2mu = mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]   # [B, T-2, H, W]
    d2y  =  y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]
    l = (d2mu - d2y).abs()
    return l.mean() if reduction=="mean" else l.sum()

def _mc_mu_var(fmodel, x, S: int, tau2: float):
    """
    MC로 예측 평균/분산 추정 → var = std^2 + tau2
    반환: (mu, var)  [B,C,H,W]
    """
    mean_pred, std_pred = _mc_mean_std(fmodel, x, S)
    var = std_pred**2 + tau2
    return mean_pred, var

def data_term_mc_dynamics(fmodel, x, y, config):
    """
    옵션 B(MC) 이분산 NLL + dynamics(vel/acc)의 합성 데이터 항
    fmodel : (적응 중인) 베이지안 모델
    x, y   : [B, Cin, H, W], [B, Cout(=T), H, W]
    config : dict (없으면 기본값 사용)
    반환   : data_loss (scalar), (mu, var) 튜플
    """
    S = int(_get(config, "MC_INNER_SAMPLES", 4))  # 기본: inner 기준
    tau2 = float(_get(config, "NLL_TAU2", 1e-4))
    w_vel = float(_get(config, "W_VEL", 0.20))
    w_acc = float(_get(config, "W_ACC", 0.05))

    # 시간 가중
    T = y.shape[1]
    time_weights = _ensure_timeweights(config, T).to(y.device, y.dtype)

    # MC로 mu,var 추정
    mu, var = _mc_mu_var(fmodel, x, S, tau2)

    # NLL
    nll = _heteroscedastic_nll_from_mu_var(mu, var, y, time_weights=time_weights)

    # dynamics 보강
    l_vel = _velocity_loss(mu, y, reduction="mean")
    l_acc = _accel_loss(mu, y, reduction="mean")

    data_loss = nll + (w_vel * l_vel) + (w_acc * l_acc)
    return data_loss, (mu, var)

# -----------------------------
# MetaLearner에 바로 꽂는 래퍼
# -----------------------------

def inner_total_loss_mc_dynamics(meta_learner, fmodel, support_x, support_y):
    """
    inner loop 목적:
      data_term(옵션B NLL + vel/acc) + beta * KL
    KL 가중치는 meta_learner.config['KL_WEIGHT'] 사용
    """
    # 데이터 항
    data_loss, _ = data_term_mc_dynamics(
        fmodel=fmodel,
        x=support_x,
        y=support_y,
        config=meta_learner.config
    )

    # KL(q||p)
    kl = fmodel.kl_divergence(meta_learner.prior_net)
    beta = float(_get(meta_learner.config, "KL_WEIGHT", 0.0))
    total = data_loss + beta * kl
    return total

def outer_data_loss_mc_dynamics(meta_learner, fmodel, query_x, query_y):
    """
    outer loop 목적:
      data_term(옵션B NLL + vel/acc)  (KL 없음)
    외부 루프에서는 일반화 성능만 측정
    """
    # 외부 루프에서는 샘플 수를 별도로 둘 수 있게 (없으면 inner와 동일)
    cfg = dict(meta_learner.config)
    cfg["MC_INNER_SAMPLES"] = int(_get(meta_learner.config, "MC_OUTER_SAMPLES", _get(meta_learner.config, "MC_INNER_SAMPLES", 4)))

    data_loss, _ = data_term_mc_dynamics(
        fmodel=fmodel,
        x=query_x,
        y=query_y,
        config=cfg
    )
    return data_loss
