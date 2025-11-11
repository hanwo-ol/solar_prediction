# meta_main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import math # isnan 확인을 위해 추가
import matplotlib.pyplot as plt

# 모듈 임포트
from meta_model import MetaLearner
from meta_dataset import SolarPredictionDataset, MetaSolarPredictionDataset, _extract_time_from_path
from meta_engine import meta_train_one_epoch, meta_evaluate
from utils import set_seed, get_device

# --- 1. 설정 (Configuration) ---
CONFIG = {
    # --- 경로 및 시드 ---
    "DATA_DIR": "/home/user/hanwool/new_npy",
    "MODEL_SAVE_PATH": "./best_bayesian_meta_model.pth",
    "SEED": 42,
    
    # --- 메타-러닝 하이퍼파라미터 ---
    "EPOCHS": 50,          # 안정적인 수렴을 위해 에포크 수를 늘립니다.
    "META_LR": 1e-8,       # 외부 학습률을 낮게 유지하여 안정적으로 학습
    "INNER_LR": 1e-7,      # [핵심 수정] 내부 학습률을 대폭 낮춥니다. (1e-2 -> 1e-4)
    "INNER_STEPS": 5,      # 내부 스텝 수도 약간 줄여서 발산 위험 감소
    "KL_WEIGHT": 1e-6,     # [핵심 수정] KL 가중치를 매우 낮춰 초반에는 MSE에 집중하도록 유도
    "GRAD_CLIP_NORM": 10.0,
    
    # --- 태스크 구성 하이퍼파라미터 ---
    "TASKS_PER_EPOCH": 100, # 더 많은 태스크를 통해 일반화 성능 향상
    "K_SHOT": 5,           # [조정] 메모리 및 안정성을 위해 줄임
    "K_QUERY": 10,         # [조정] 메모리 및 안정성을 위해 줄임
    
    # --- 평가 하이퍼파라미터 ---
    "NUM_ADAPTATION_STEPS": 10,
    "NUM_EVAL_SAMPLES": 10,

    # --- 데이터 및 모델 기본 설정 ---
    "INPUT_LEN": 4,
    "TARGET_LEN": 4,
    "DATA_MIN": 0.0,
    "DATA_MAX": 26.41
}

# visualize_meta_predictions 함수는 이전과 동일하게 유지
def visualize_meta_predictions(mean_pred, std_pred, ground_truth, sample_idx=0):
    """
    [수정됨] 메타-러닝 평가 결과를 시각화하는 완전한 함수.
    평균 예측, 불확실성(표준편차), 실제 값을 비교합니다.
    """
    print("\n--- Visualizing Final Test Task Prediction ---")
    
    # -1~1 범위를 원래 데이터 범위로 되돌리는 함수
    def denormalize(tensor):
        val_range = CONFIG['DATA_MAX'] - CONFIG['DATA_MIN']
        return (tensor.cpu().numpy() * (val_range / 2.0)) + ((CONFIG['DATA_MAX'] + CONFIG['DATA_MIN']) / 2.0)

    # 텐서에서 시각화할 샘플 선택 (배치의 첫 번째 아이템)
    gt_sequence = denormalize(ground_truth[sample_idx])
    mean_sequence = denormalize(mean_pred[sample_idx])
    std_sequence = std_pred[sample_idx].cpu().numpy() # 표준편차는 정규화 해제 불필요

    num_steps = CONFIG['TARGET_LEN']
    fig, axes = plt.subplots(3, num_steps, figsize=(num_steps * 4, 10))
    fig.suptitle(f'Test Task Adaptation Result (Sample {sample_idx+1})\nPrediction, Uncertainty, and Ground Truth', fontsize=16)

    for j in range(num_steps):
        time_step = 30 * (j + 1)
        
        # 1행: Ground Truth
        ax = axes[0, j]
        im = ax.imshow(gt_sequence[j], cmap='gray', vmin=CONFIG['DATA_MIN'], vmax=CONFIG['DATA_MAX'])
        ax.set_title(f'Target (t+{time_step}m)')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 2행: Mean Prediction
        ax = axes[1, j]
        im = ax.imshow(mean_sequence[j], cmap='gray', vmin=CONFIG['DATA_MIN'], vmax=CONFIG['DATA_MAX'])
        ax.set_title(f'Mean Prediction (t+{time_step}m)')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 3행: Uncertainty (Standard Deviation)
        ax = axes[2, j]
        # 불확실성은 다른 컬러맵을 사용하여 명확히 구분
        im = ax.imshow(std_sequence[j], cmap='viridis') 
        ax.set_title(f'Uncertainty (Std Dev)')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_path = f"meta_prediction_sample_{sample_idx+1}.png"
    plt.savefig(save_path)
    print(f"Saved meta-prediction visualization to {save_path}")
    plt.show()

def main():
    set_seed(CONFIG['SEED'])
    device = get_device()

    # --- 2. 데이터 준비 ---
    # ... (이전 답변의 데이터 준비 코드와 동일) ...
    data_dir = Path(CONFIG['DATA_DIR'])
    all_files = sorted(list(data_dir.glob("*.npy")))
    if not all_files:
        raise FileNotFoundError(f"Error: No .npy files found in {data_dir}.")
        
    train_files = [p for p in all_files if _extract_time_from_path(p) and _extract_time_from_path(p).year in [2021, 2022]]
    val_files = [p for p in all_files if _extract_time_from_path(p) and _extract_time_from_path(p).year == 2023]

    transform = transforms.Lambda(
        lambda x: (x - (CONFIG['DATA_MAX'] + CONFIG['DATA_MIN']) / 2.0) / ((CONFIG['DATA_MAX'] - CONFIG['DATA_MIN']) / 2.0)
    )

    base_train_dataset = SolarPredictionDataset(train_files, CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'], transform)
    base_val_dataset = SolarPredictionDataset(val_files, CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'], transform)

    meta_train_dataset = MetaSolarPredictionDataset(base_train_dataset, CONFIG['TASKS_PER_EPOCH'], CONFIG['K_SHOT'], CONFIG['K_QUERY'])
    meta_train_loader = DataLoader(meta_train_dataset, batch_size=1, shuffle=True, num_workers=2)

    val_task_generator = MetaSolarPredictionDataset(base_val_dataset, 1, CONFIG['K_SHOT'], CONFIG['K_QUERY'])
    val_task = val_task_generator[0]
    print(f"Validation task created with {val_task[0].shape[0]} support samples and {val_task[2].shape[0]} query samples.")

    # --- 3. 모델, 옵티마이저, 스케줄러 정의 ---
    meta_learner = MetaLearner(CONFIG).to(device)
    meta_optimizer = optim.AdamW(meta_learner.parameters(), lr=CONFIG['META_LR'])
    
    # --- 안정화 장치: 학습률 스케줄러 ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, 'min', patience=3, factor=0.5)

    # --- 4. 메타-훈련 루프 (수정됨) ---
    best_val_loss = float('inf')
    print("\n--- Starting Meta-Training ---")
    for epoch in range(CONFIG['EPOCHS']):
        
        train_loss = meta_train_one_epoch(
            meta_learner, 
            meta_train_loader, 
            meta_optimizer, 
            device, 
            CONFIG['GRAD_CLIP_NORM']
        )
        
        # --- 매 에포크 검증 (디버깅 모드 활성화) ---
        # 첫 에포크 또는 nan 발생 시에만 디버깅 로그를 출력하도록 설정 가능
        is_debug_epoch = (epoch == 0) 

        val_loss, _, _, _ = meta_evaluate(
            meta_learner, 
            val_task, 
            device, 
            CONFIG['NUM_ADAPTATION_STEPS'], 
            CONFIG['NUM_EVAL_SAMPLES'],
            debug=is_debug_epoch # 디버그 플래그 전달
        )
        
        print(f"\nEpoch {epoch+1}/{CONFIG['EPOCHS']} | Meta-Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # --- nan 값 처리 및 모델 저장 ---
        # val_loss가 유효한 숫자인지 확인
        if not math.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(meta_learner.state_dict(), CONFIG['MODEL_SAVE_PATH'])
            print(f"Best model saved to {CONFIG['MODEL_SAVE_PATH']} with validation loss: {best_val_loss:.6f}")

        # --- 스케줄러 업데이트 ---
        if not math.isnan(val_loss):
            scheduler.step(val_loss)

    print("\n--- Meta-Training Finished ---")

    # --- 5. 최종 평가 및 시각화 ---
    print("\n--- Final Evaluation on a New Test Task ---")
    
    # 모델 파일이 존재하는지 확인 후 로드
    if Path(CONFIG['MODEL_SAVE_PATH']).exists():
        meta_learner.load_state_dict(torch.load(CONFIG['MODEL_SAVE_PATH']))
        print("Loaded best model for final evaluation.")
        
        test_task_generator = MetaSolarPredictionDataset(base_val_dataset, 1, CONFIG['K_SHOT'], CONFIG['K_QUERY'])
        test_task = test_task_generator[0]
        
        test_loss, mean_pred, std_pred, ground_truth = meta_evaluate(
            meta_learner, 
            test_task, 
            device, 
            CONFIG['NUM_ADAPTATION_STEPS'], 
            CONFIG['NUM_EVAL_SAMPLES']
        )
        
        # --- [핵심 수정] 반환 값 유효성 검사 ---
        if not math.isnan(test_loss) and mean_pred is not None:
            print(f"Final Test Task Loss: {test_loss:.6f}")
            visualize_meta_predictions(mean_pred, std_pred, ground_truth, sample_idx=0)
        else:
            print("Final evaluation failed: NaN was produced during adaptation.")
            
    else:
        print("Could not find a saved model. Final evaluation skipped.")

if __name__ == '__main__':
    main()