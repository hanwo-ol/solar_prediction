
import torch
import torch.nn.functional as F
from tqdm import tqdm
import higher
from torch.amp import autocast, GradScaler

@torch.no_grad()
def _flatten_task_batch(x, is_target=False):
    if x.dim() == 5:
        b, k, c, h, w = x.shape
        assert b == 1, f"Expected loader batch=1, got {b}"
        return x.view(k, c, h, w)
    elif x.dim() == 4:
        return x
    else:
        raise RuntimeError(f"Unexpected tensor shape {tuple(x.shape)} for {'target' if is_target else 'input'}")

def meta_train_one_epoch(meta_learner, dataloader, meta_optimizer, device, grad_clip_norm):
    """Meta-train one epoch. Returns (avg_outer_loss, avg_metrics). avg_metrics computed on query set."""
    meta_learner.train()
    cfg = meta_learner.config
    inner_steps = int(cfg.get("NUM_ADAPTATION_STEPS", cfg.get("INNER_STEPS", 5)))
    inner_lr    = float(cfg.get("INNER_LR", 1e-3))
    use_amp     = bool(cfg.get("USE_AMP", False))
    scaler = GradScaler('cuda', enabled=use_amp)

    running_outer = 0.0
    running_metrics = {"mse":0.0, "mae":0.0, "ssim":0.0}
    n_tasks = 0

    meta_optimizer.zero_grad(set_to_none=True)

    for batch in tqdm(dataloader, desc="Meta-Training"):
        if len(batch) != 4:
            raise ValueError("Each batch must be (support_x, support_y, query_x, query_y)." )
        support_x, support_y, query_x, query_y = batch

        support_x = support_x.to(device, non_blocking=True)
        support_y = support_y.to(device, non_blocking=True)
        query_x   = query_x.to(device, non_blocking=True)
        query_y   = query_y.to(device, non_blocking=True)

        support_x = _flatten_task_batch(support_x, is_target=False).contiguous()
        support_y = _flatten_task_batch(support_y, is_target=True).contiguous()
        query_x   = _flatten_task_batch(query_x,   is_target=False).contiguous()
        query_y   = _flatten_task_batch(query_y,   is_target=True).contiguous()

        inner_opt = torch.optim.SGD(meta_learner.prior_net.parameters(), lr=inner_lr)

        with higher.innerloop_ctx(
            meta_learner.prior_net, inner_opt,
            copy_initial_weights=True, track_higher_grads=True
        ) as (fmodel, diffopt):

            for _ in range(inner_steps):
                with autocast('cuda', enabled=use_amp):
                    inner_loss = meta_learner.inner_loop_loss(fmodel, support_x, support_y)
                    inner_loss = torch.nan_to_num(inner_loss, nan=0.0, posinf=1e6, neginf=1e6)
                diffopt.step(inner_loss)

            with autocast('cuda', enabled=use_amp):
                outer_loss = meta_learner.outer_loop_loss(fmodel, query_x, query_y)
                outer_loss = torch.nan_to_num(outer_loss, nan=0.0, posinf=1e6, neginf=1e6)

            if use_amp:
                scaler.scale(outer_loss).backward()
            else:
                outer_loss.backward()

            running_outer += float(outer_loss.detach().cpu())
            n_tasks += 1

            with torch.no_grad():
                S = int(meta_learner.config.get("MC_OUTER_SAMPLES", meta_learner.config.get("MC_INNER_SAMPLES", 4)))
                preds = []
                for _ in range(S):
                    p = fmodel.prior_net(query_x, sample=True)
                    p = torch.nan_to_num(p, nan=0.0, posinf=1e3, neginf=-1e3)
                    preds.append(p)
                mean_pred = torch.stack(preds,0).mean(0)
                from utils import compute_metrics
                data_range = (meta_learner.config.get("DATA_MAX", 1.0) - meta_learner.config.get("DATA_MIN", 0.0)) or 1.0
                mt = compute_metrics(mean_pred, query_y, data_range=data_range)
                for k in running_metrics: running_metrics[k] += mt[k]

    if grad_clip_norm is not None and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(meta_learner.parameters(), grad_clip_norm)

    if use_amp:
        scaler.step(meta_optimizer); scaler.update()
    else:
        meta_optimizer.step()
    meta_optimizer.zero_grad(set_to_none=True)

    denom = max(1, n_tasks)
    avg_outer = running_outer / denom
    for k in running_metrics: running_metrics[k] /= denom
    return avg_outer, running_metrics

@torch.no_grad()
def meta_evaluate(meta_learner, task, device, num_adaptation_steps, num_eval_samples):
    """Returns: (test_loss, mean_pred, std_pred, query_y, metrics)."""
    meta_learner.eval()
    support_x, support_y, query_x, query_y = task

    support_x = support_x.to(device, non_blocking=True)
    support_y = support_y.to(device, non_blocking=True)
    query_x   = query_x.to(device, non_blocking=True)
    query_y   = query_y.to(device, non_blocking=True)

    support_x = _flatten_task_batch(support_x, is_target=False).contiguous()
    support_y = _flatten_task_batch(support_y, is_target=True).contiguous()
    query_x   = _flatten_task_batch(query_x,   is_target=False).contiguous()
    query_y   = _flatten_task_batch(query_y,   is_target=True).contiguous()

    inner_lr = float(meta_learner.config.get("INNER_LR", 1e-3))
    inner_opt = torch.optim.SGD(meta_learner.prior_net.parameters(), lr=inner_lr)
    use_amp = bool(meta_learner.config.get("USE_AMP", False))

    with higher.innerloop_ctx(
        meta_learner.prior_net, inner_opt,
        copy_initial_weights=True, track_higher_grads=False
    ) as (fmodel, diffopt):
        for _ in range(num_adaptation_steps):
            with autocast('cuda', enabled=use_amp):
                inner_loss = meta_learner.inner_loop_loss(fmodel, support_x, support_y)
                inner_loss = torch.nan_to_num(inner_loss, nan=0.0, posinf=1e6, neginf=1e6)
            diffopt.step(inner_loss)

        predictions = []
        for _ in range(num_eval_samples):
            pred = fmodel.prior_net(query_x, sample=True)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3)
            predictions.append(pred.cpu())

    predictions_tensor = torch.stack(predictions)
    mean_prediction = predictions_tensor.mean(dim=0)
    std_prediction = predictions_tensor.std(dim=0)
    test_loss = F.mse_loss(mean_prediction, query_y.cpu())

    from utils import compute_metrics
    data_range = (meta_learner.config.get("DATA_MAX", 1.0) - meta_learner.config.get("DATA_MIN", 0.0)) or 1.0
    metrics = compute_metrics(mean_prediction, query_y.cpu(), data_range=data_range)
    return test_loss.item(), mean_prediction, std_prediction, query_y.cpu(), metrics
