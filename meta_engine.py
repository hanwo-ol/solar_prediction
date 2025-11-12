import torch
import torch.nn.functional as F
from tqdm import tqdm
import higher
import numpy as np
import copy

def meta_train_one_epoch(meta_learner, dataloader, meta_optimizer, device, grad_clip_norm):
    """
    한 에폭의 메타-학습을 수행하고 평균 outer loss를 반환.
    - higher로 meta_learner.prior_net을 inner-loop 적응
    - NaN/Inf 가드 및 gradient clipping 적용
    - AMP(선택) 지원: CONFIG['USE_AMP']=True 시 자동 활성화
    - 🔧 입력 형태 [1, K, C, H, W] -> [B=K, C, H, W]로 평탄화 처리
    """
    import torch
    import higher
    from torch.amp import autocast, GradScaler  # 최신 권장 API

    def _flatten_task_batch(x, is_target=False):
        """
        x: [1, K, C, H, W] -> [K, C, H, W]
        (로더 batch_size=1 전제. 만약 [K, C, H, W]면 그대로 통과)
        """
        if x.dim() == 5:
            b, k, c, h, w = x.shape
            assert b == 1, f"Expected loader batch=1, got {b}"
            return x.view(k, c, h, w)
        elif x.dim() == 4:
            return x
        else:
            raise RuntimeError(f"Unexpected tensor shape {tuple(x.shape)} for {'target' if is_target else 'input'}")

    meta_learner.train()
    cfg = meta_learner.config

    # 설정 키 호환 (INNER_STEPS vs NUM_ADAPTATION_STEPS)
    inner_steps = int(cfg.get("NUM_ADAPTATION_STEPS", cfg.get("INNER_STEPS", 5)))
    inner_lr    = float(cfg.get("INNER_LR", 1e-3))
    use_amp     = bool(cfg.get("USE_AMP", False))

    scaler = GradScaler('cuda', enabled=use_amp)

    running_outer = 0.0
    n_tasks = 0

    meta_optimizer.zero_grad(set_to_none=True)

    for batch in tqdm(dataloader, desc="Meta-Training"):
        if len(batch) != 4:
            raise ValueError("Each batch must be (support_x, support_y, query_x, query_y).")
        support_x, support_y, query_x, query_y = batch

        # 디바이스 이동
        support_x = support_x.to(device, non_blocking=True)
        support_y = support_y.to(device, non_blocking=True)
        query_x   = query_x.to(device, non_blocking=True)
        query_y   = query_y.to(device, non_blocking=True)

        # 🔧 [1, K, C, H, W] -> [K, C, H, W] 평탄화
        support_x = _flatten_task_batch(support_x, is_target=False).contiguous()
        support_y = _flatten_task_batch(support_y, is_target=True).contiguous()
        query_x   = _flatten_task_batch(query_x,   is_target=False).contiguous()
        query_y   = _flatten_task_batch(query_y,   is_target=True).contiguous()

        # inner optimizer (task별 초기화)
        base_params = list(meta_learner.prior_net.parameters())
        inner_opt = torch.optim.SGD(base_params, lr=inner_lr, momentum=0.0)

        # higher 컨텍스트: prior_net을 적응 대상으로 사용
        with higher.innerloop_ctx(
            meta_learner.prior_net,
            inner_opt,
            copy_initial_weights=True,     # 태스크마다 깨끗한 초기 파라미터
            track_higher_grads=True        # 메타 업데이트 위해 고계 미분 추적
        ) as (fmodel, diffopt):

            # ---------- Inner loop (support) ----------
            for _ in range(inner_steps):
                with autocast('cuda', enabled=use_amp):
                    inner_loss = meta_learner.inner_loop_loss(fmodel, support_x, support_y)
                    inner_loss = torch.nan_to_num(inner_loss, nan=0.0, posinf=1e6, neginf=1e6)

                # higher가 내부에서 backward+update 처리. inner에서는 .grad를 건드리지 않음
                diffopt.step(inner_loss)

            # ---------- Outer loss (query) ----------
            with autocast('cuda', enabled=use_amp):
                outer_loss = meta_learner.outer_loop_loss(fmodel, query_x, query_y)
                outer_loss = torch.nan_to_num(outer_loss, nan=0.0, posinf=1e6, neginf=1e6)

            # 메타 그라드 누적 (원본 prior_net 파라미터로)
            if use_amp:
                scaler.scale(outer_loss).backward()
            else:
                outer_loss.backward()

            running_outer += float(outer_loss.detach().cpu())
            n_tasks += 1

        # ---------- Meta step (task마다 업데이트) ----------
        # 메타 그라드 가드 + 클립
        for p in meta_learner.prior_net.parameters():
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(meta_learner.prior_net.parameters(), max_norm=grad_clip_norm)

        if use_amp:
            scaler.step(meta_optimizer)
            scaler.update()
        else:
            meta_optimizer.step()

        meta_optimizer.zero_grad(set_to_none=True)

    avg_outer = running_outer / max(1, n_tasks)
    return avg_outer




def meta_evaluate(meta_learner, test_task, device, num_adaptation_steps, num_eval_samples, debug=False):
    """
    평가 함수: 현재는 MSE 기반(원 레포 스타일 유지).
    """
    meta_learner.eval()
    
    support_x, support_y, query_x, query_y = test_task
    support_x, support_y = support_x.to(device), support_y.to(device)
    query_x, query_y = query_x.to(device), query_y.to(device)

    fmodel = copy.deepcopy(meta_learner)
    fmodel.to(device)
    
    inner_opt = torch.optim.SGD(fmodel.parameters(), lr=meta_learner.config['INNER_LR'])
    
    for step in range(num_adaptation_steps):
        inner_opt.zero_grad()
        fmodel.prior_net.train()
        outputs = fmodel.prior_net(support_x, sample=True)
        mse_loss = F.mse_loss(outputs, support_y)
        kl_loss = fmodel.prior_net.kl_divergence(meta_learner.prior_net)
        inner_loss = mse_loss + meta_learner.config['KL_WEIGHT'] * kl_loss

        if torch.isnan(inner_loss) or torch.isinf(inner_loss):
            print(f"!!! NaN or Inf detected at adaptation step {step+1}. Stopping evaluation. !!!")
            return float('nan'), None, None, None
            
        inner_loss.backward()
        inner_opt.step()
        
    fmodel.eval()
    with torch.no_grad():
        predictions = []
        for _ in range(num_eval_samples):
            pred = fmodel.prior_net(query_x, sample=True)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3)
            predictions.append(pred.cpu())
        
        predictions_tensor = torch.stack(predictions)
        mean_prediction = predictions_tensor.mean(dim=0)
        std_prediction = predictions_tensor.std(dim=0)
        test_loss = F.mse_loss(mean_prediction, query_y.cpu())

    del fmodel
    torch.cuda.empty_cache()

    return test_loss.item(), mean_prediction, std_prediction, query_y.cpu()
