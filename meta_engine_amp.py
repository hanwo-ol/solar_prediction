# meta_engine_amp.py
# ---------------------------------------------------------------------------------
# AMP-enabled meta-training / evaluation engine (tuple/dict task both supported)
# ---------------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import higher


# -----------------------------
# 유틸: 배치 평탄화 & 디바이스 이동
# -----------------------------
def _squeeze5d(x: torch.Tensor) -> torch.Tensor:
    # 기대 입력: [1, K, C, H, W] 또는 [K, C, H, W]
    if isinstance(x, torch.Tensor) and x.dim() == 5 and x.size(0) == 1:
        return x.squeeze(0)
    return x

def _unpack_task(task) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    task가 dict 또는 (support_x, support_y, query_x, query_y) 튜플/리스트일 수 있음.
    """
    if isinstance(task, dict):
        return task["support_x"], task["support_y"], task["query_x"], task["query_y"]
    if isinstance(task, (list, tuple)) and len(task) == 4:
        return task[0], task[1], task[2], task[3]
    raise TypeError(f"Unsupported task type: {type(task)}. Expect dict or 4-tuple.")

def _move_and_flatten(task, device: torch.device):
    sx, sy, qx, qy = _unpack_task(task)
    sx = _squeeze5d(sx).to(device, non_blocking=True)  # [K,C,H,W]
    sy = _squeeze5d(sy).to(device, non_blocking=True)
    qx = _squeeze5d(qx).to(device, non_blocking=True)
    qy = _squeeze5d(qy).to(device, non_blocking=True)
    return sx, sy, qx, qy


# -----------------------------
# 메타-트레인 1 에폭
# -----------------------------
def meta_train_one_epoch(
    meta_learner: nn.Module,
    meta_train_loader,
    meta_optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    config: Dict[str, Any],
):
    meta_learner.train()
    meta_learner.to(device)

    use_amp = bool(config.get("USE_AMP", True))
    amp_dtype = torch.bfloat16 if config.get("AMP_DTYPE", "bf16") == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    inner_steps = int(config.get("INNER_STEPS", config.get("NUM_ADAPTATION_STEPS", 5)))
    inner_lr = float(config.get("INNER_LR", 1e-6))

    running_outer = 0.0
    n_tasks = 0

    pbar = tqdm(meta_train_loader, desc="Meta-Training", total=len(meta_train_loader))
    for task in pbar:
        support_x, support_y, query_x, query_y = _move_and_flatten(task, device)

        # higher: 모델 사본 + 미분가능 inner optimizer
        base_inner_optim = torch.optim.SGD(meta_learner.prior_net.parameters(), lr=inner_lr)
        with higher.innerloop_ctx(
            meta_learner.prior_net, base_inner_optim,
            copy_initial_weights=True, track_higher_grads=True
        ) as (fmodel, diffopt):

            # ----- inner loop -----
            for _ in range(inner_steps):
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    inner_loss = meta_learner.inner_loop_loss(fmodel, support_x, support_y)
                diffopt.step(inner_loss)  # 내부에서 backward 호출

            # ----- outer loop -----
            meta_optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outer_loss = meta_learner.outer_loop_loss(fmodel, query_x, query_y)

            if not torch.isfinite(outer_loss):
                print("!!! [warn] outer_loss is NaN/Inf. Skip this task.")
                continue

            if scaler.is_enabled():
                scaler.scale(outer_loss).backward()
                scaler.unscale_(meta_optimizer)
                clip_grad_norm_(meta_learner.parameters(), max_norm=grad_clip_norm)
                scaler.step(meta_optimizer)
                scaler.update()
            else:
                outer_loss.backward()
                clip_grad_norm_(meta_learner.parameters(), max_norm=grad_clip_norm)
                meta_optimizer.step()

            running_outer += float(outer_loss.detach().cpu())
            n_tasks += 1
            pbar.set_postfix_str(f"outer={running_outer/max(n_tasks,1):.4f}")

        del support_x, support_y, query_x, query_y, inner_loss, outer_loss, fmodel, diffopt, base_inner_optim
        torch.cuda.empty_cache()        

    return running_outer / max(n_tasks, 1)


# -----------------------------
# 메타-평가 (validation/test)
# -----------------------------
@torch.no_grad()
def meta_evaluate(
    meta_learner: nn.Module,
    meta_val_loader,
    device: torch.device,
    config: Dict[str, Any],
):
    meta_learner.eval()
    meta_learner.to(device)

    use_amp = bool(config.get("USE_AMP", True))
    amp_dtype = torch.bfloat16 if config.get("AMP_DTYPE", "bf16") == "bf16" else torch.float16

    inner_steps = int(config.get("INNER_STEPS", config.get("NUM_ADAPTATION_STEPS", 5)))
    inner_lr = float(config.get("INNER_LR", 1e-6))

    running = 0.0
    n_tasks = 0

    for task in tqdm(meta_val_loader, desc="Meta-Evaluate", total=len(meta_val_loader)):
        support_x, support_y, query_x, query_y = _move_and_flatten(task, device)

        base_inner_optim = torch.optim.SGD(meta_learner.prior_net.parameters(), lr=inner_lr)
        with higher.innerloop_ctx(
            meta_learner.prior_net, base_inner_optim,
            copy_initial_weights=True, track_higher_grads=False
        ) as (fmodel, diffopt):

            for _ in range(inner_steps):
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    inner_loss = meta_learner.inner_loop_loss(fmodel, support_x, support_y)
                diffopt.step(inner_loss)

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outer_loss = meta_learner.outer_loop_loss(fmodel, query_x, query_y)

        running += float(outer_loss.detach().cpu())
        
        n_tasks += 1

    return running / max(n_tasks, 1)
