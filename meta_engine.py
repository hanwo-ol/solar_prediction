# meta_engine.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import higher
import numpy as np

def meta_train_one_epoch(meta_learner, dataloader, meta_optimizer, device):
    """
    한 에포크 동안 메타-러닝 훈련을 수행합니다.
    """
    meta_learner.train()
    total_meta_loss = 0.0

    # 데이터로더는 (support_x, support_y, query_x, query_y) 튜플을 반환
    for task_batch in tqdm(dataloader, desc="Meta-Training"):
        support_x, support_y, query_x, query_y = task_batch
        
        # 메타-배치 사이즈가 1이라고 가정 (일반적으로 메타-러닝에서 사용)
        # (1, k_shot, C, H, W) -> (k_shot, C, H, W)
        support_x, support_y = support_x.squeeze(0).to(device), support_y.squeeze(0).to(device)
        query_x, query_y = query_x.squeeze(0).to(device), query_y.squeeze(0).to(device)

        meta_optimizer.zero_grad()
        
        # higher 라이브러리를 사용하여 상태 저장 없이 미분 가능한 옵티마이저 생성
        # 내부 루프에서는 사전 분포(prior_net)의 파라미터를 복사하여 적응시킴
        inner_opt = torch.optim.SGD(meta_learner.prior_net.parameters(), lr=meta_learner.config['INNER_LR'])
        
        with higher.innerloop_ctx(meta_learner.prior_net, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):
            # 1. Inner loop: 서포트 셋으로 fmodel(사후 모델)을 적응시킴
            for _ in range(meta_learner.config['INNER_STEPS']):
                # meta_learner의 inner_loop_loss는 fmodel(사후)과 self.prior_net(사전)을 모두 사용
                inner_loss = meta_learner.inner_loop_loss(fmodel, support_x, support_y)
                diffopt.step(inner_loss)

            # 2. Outer loop: 적응된 fmodel을 쿼리 셋으로 평가
            outer_loss = meta_learner.outer_loop_loss(fmodel, query_x, query_y)
            
            # 3. Meta-gradient 계산 및 역전파
            # outer_loss의 그래디언트가 inner_loop의 연산을 거슬러 올라가
            # 원본 meta_learner.prior_net의 파라미터에 대해 계산됨
            outer_loss.backward()

        # 4. Meta-parameter (사전 분포 파라미터) 업데이트
        meta_optimizer.step()
        
        total_meta_loss += outer_loss.item()

    return total_meta_loss / len(dataloader)

def meta_evaluate(meta_learner, test_task, device, num_adaptation_steps, num_eval_samples):
    """
    새로운 단일 테스트 태스크에 대해 모델을 적응시키고,
    여러 번의 샘플링을 통해 예측과 불확실성을 평가합니다.
    """
    meta_learner.eval()
    
    support_x, support_y, query_x, query_y = test_task
    # 배치 차원이 없으므로 squeeze 불필요
    support_x, support_y = support_x.to(device), support_y.to(device)
    query_x, query_y = query_x.to(device), query_y.to(device)

    # higher를 사용하여 테스트 시 모델 적응
    inner_opt = torch.optim.SGD(meta_learner.prior_net.parameters(), lr=meta_learner.config['INNER_LR'])
    
    # fmodel이 최종적으로 적응된 모델이 됨
    with higher.innerloop_ctx(meta_learner.prior_net, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):
        for _ in range(num_adaptation_steps):
            inner_loss = meta_learner.inner_loop_loss(fmodel, support_x, support_y)
            diffopt.step(inner_loss)
        
        # 평가: 적응된 fmodel(사후 분포)에서 여러 번 샘플링하여 예측
        with torch.no_grad():
            predictions = []
            for _ in range(num_eval_samples):
                # sample=True로 설정하여 베이지안 레이어에서 가중치를 샘플링
                pred = fmodel(query_x, sample=True)
                predictions.append(pred.cpu())
            
            # (num_samples, k_query, C, H, W)
            predictions_tensor = torch.stack(predictions)
            
            # 예측의 평균과 표준편차 계산
            mean_prediction = predictions_tensor.mean(dim=0)
            std_prediction = predictions_tensor.std(dim=0)
            
            # 최종 손실은 평균 예측값을 기준으로 계산
            test_loss = F.mse_loss(mean_prediction, query_y.cpu())

    return test_loss.item(), mean_prediction, std_prediction, query_y.cpu()