# meta_engine.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import higher
import numpy as np
import copy

def meta_train_one_epoch(meta_learner, dataloader, meta_optimizer, device, grad_clip_norm):
    """
    [수정됨] 그래디언트 클리핑이 추가된 메타-러닝 훈련 함수.
    """
    meta_learner.train()
    total_meta_loss = 0.0

    for task_batch in tqdm(dataloader, desc="Meta-Training"):
        support_x, support_y, query_x, query_y = task_batch
        support_x, support_y = support_x.squeeze(0).to(device), support_y.squeeze(0).to(device)
        query_x, query_y = query_x.squeeze(0).to(device), query_y.squeeze(0).to(device)

        meta_optimizer.zero_grad()
        
        inner_opt = torch.optim.SGD(meta_learner.prior_net.parameters(), lr=meta_learner.config['INNER_LR'])
        
        with higher.innerloop_ctx(meta_learner.prior_net, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):
            for _ in range(meta_learner.config['INNER_STEPS']):
                inner_loss = meta_learner.inner_loop_loss(fmodel, support_x, support_y)
                # inner_loss에 nan이 있는지 확인 (디버깅용)
                if torch.isnan(inner_loss):
                    print("Warning: NaN detected in inner loop loss.")
                    continue # 이 태스크는 건너뜀
                diffopt.step(inner_loss)

            outer_loss = meta_learner.outer_loop_loss(fmodel, query_x, query_y)
            if torch.isnan(outer_loss):
                print("Warning: NaN detected in outer loop loss.")
                continue # 이 태스크는 건너뜀
            
            outer_loss.backward()

        # --- 안정화 장치: 그래디언트 클리핑 ---
        torch.nn.utils.clip_grad_norm_(meta_learner.parameters(), grad_clip_norm)
        
        meta_optimizer.step()
        total_meta_loss += outer_loss.item()

    return total_meta_loss / len(dataloader)

def meta_evaluate(meta_learner, test_task, device, num_adaptation_steps, num_eval_samples, debug=False):
    """
    [수정됨] nan 추적을 위한 디버깅 기능이 추가된 평가 함수.
    """
    meta_learner.eval()
    
    support_x, support_y, query_x, query_y = test_task
    support_x, support_y = support_x.to(device), support_y.to(device)
    query_x, query_y = query_x.to(device), query_y.to(device)

    fmodel = copy.deepcopy(meta_learner)
    fmodel.to(device)
    
    inner_opt = torch.optim.SGD(fmodel.parameters(), lr=meta_learner.config['INNER_LR'])
    
    if debug:
        print("\n--- Starting Debugging in meta_evaluate ---")

    for step in range(num_adaptation_steps):
        inner_opt.zero_grad()
        
        # inner_loop_loss를 구성하는 각 요소를 따로 계산하여 추적
        fmodel.prior_net.train() # BN, Dropout 등을 훈련 모드로 설정
        outputs = fmodel.prior_net(support_x, sample=True)
        mse_loss = F.mse_loss(outputs, support_y)
        kl_loss = fmodel.prior_net.kl_divergence(meta_learner.prior_net)
        inner_loss = mse_loss + meta_learner.config['KL_WEIGHT'] * kl_loss

        if debug:
            print(f"  [Adaptation Step {step+1}/{num_adaptation_steps}]")
            print(f"    MSE Loss: {mse_loss.item():.6f}")
            print(f"    KL Loss: {kl_loss.item():.6f}")
            print(f"    Total Inner Loss: {inner_loss.item():.6f}")

        if torch.isnan(inner_loss) or torch.isinf(inner_loss):
            print(f"!!! NaN or Inf detected at adaptation step {step+1}. Stopping evaluation. !!!")
            if debug:
                # 파라미터 값 확인
                for name, param in fmodel.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"    - NaN/Inf found in parameter: {name}")
            return float('nan'), None, None, None
            
        inner_loss.backward()

        # 그래디언트 크기 확인
        if debug:
            total_norm = 0
            for p in fmodel.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"    Gradient Norm: {total_norm:.6f}")

        inner_opt.step()
        
    # 평가 시에는 eval 모드로 전환
    fmodel.eval()
    with torch.no_grad():
        predictions = []
        for _ in range(num_eval_samples):
            pred = fmodel.prior_net(query_x, sample=True)
            predictions.append(pred.cpu())
        
        predictions_tensor = torch.stack(predictions)
        mean_prediction = predictions_tensor.mean(dim=0)
        std_prediction = predictions_tensor.std(dim=0)
        test_loss = F.mse_loss(mean_prediction, query_y.cpu())

    del fmodel
    torch.cuda.empty_cache()

    return test_loss.item(), mean_prediction, std_prediction, query_y.cpu()