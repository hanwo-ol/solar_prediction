# meta_main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# 메타-러닝을 위해 새로 작성된 모듈들을 임포트합니다.
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
    "EPOCHS": 5,          # 메타-러닝은 더 많은 에포크가 필요할 수 있습니다.
    "META_LR": 1e-4,       # 외부 루프(사전 분포) 학습률
    "INNER_LR": 1e-2,      # 내부 루프(사후 분포 적응) 학습률
    "INNER_STEPS": 5,      # 내부 루프 적응 횟수
    "KL_WEIGHT": 0.01,     # ELBO 손실에서 KL Divergence의 가중치
    
    # --- 태스크 구성 하이퍼파라미터 ---
    "TASKS_PER_EPOCH": 10,# 한 에포크당 생성할 랜덤 태스크의 수
    "K_SHOT": 10,          # 서포트 셋(적응용) 샘플 수
    "K_QUERY": 15,         # 쿼리 셋(평가용) 샘플 수
    
    # --- 평가 하이퍼파라미터 ---
    "NUM_ADAPTATION_STEPS": 10, # 평가 시 적응 횟수
    "NUM_EVAL_SAMPLES": 20,     # 불확실성 측정을 위한 샘플링 횟수

    # --- 데이터 및 모델 기본 설정 ---
    "INPUT_LEN": 4,
    "TARGET_LEN": 4,
    "DATA_MIN": 0.0,
    "DATA_MAX": 26.41
}

def visualize_meta_predictions(mean_pred, std_pred, ground_truth, sample_idx=0):
    """
    메타-평가 결과를 시각화합니다. (평균 예측, 불확실성, 실제값, 오차)
    """
    print(f"\n--- Visualizing Meta-Prediction for Sample {sample_idx+1} ---")
    
    # -1~1 범위를 원래 데이터 범위로 되돌리는 함수
    def denormalize(tensor):
        val_range = CONFIG['DATA_MAX'] - CONFIG['DATA_MIN']
        return (tensor.cpu() * (val_range / 2.0)) + ((CONFIG['DATA_MAX'] + CONFIG['DATA_MIN']) / 2.0)

    # 쿼리 셋의 첫 번째 샘플을 시각화 대상으로 선택
    mean_pred_sample = denormalize(mean_pred[0])
    std_pred_sample = std_pred[0].cpu() # 표준편차는 정규화 해제 불필요
    ground_truth_sample = denormalize(ground_truth[0])

    fig, axes = plt.subplots(4, CONFIG['TARGET_LEN'], figsize=(CONFIG['TARGET_LEN'] * 4, 13))
    fig.suptitle(f'Meta-Prediction Sample {sample_idx+1}\n(Predicting t+30m to t+120m after adaptation)', fontsize=16)

    for j in range(CONFIG['TARGET_LEN']):
        time_step = 30 * (j + 1)
        
        # 1행: Ground Truth
        ax = axes[0, j]
        im = ax.imshow(ground_truth_sample[j], cmap='gray', vmin=CONFIG['DATA_MIN'], vmax=CONFIG['DATA_MAX'])
        ax.set_title(f'Target (t+{time_step}m)')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 2행: Mean Prediction (최종 예측)
        ax = axes[1, j]
        im = ax.imshow(mean_pred_sample[j], cmap='gray', vmin=CONFIG['DATA_MIN'], vmax=CONFIG['DATA_MAX'])
        ax.set_title(f'Mean Prediction (t+{time_step}m)')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 3행: Uncertainty (Standard Deviation)
        ax = axes[2, j]
        im = ax.imshow(std_pred_sample[j], cmap='viridis') # 불확실성은 다른 컬러맵 사용
        ax.set_title(f'Uncertainty (Std Dev)')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 4행: Difference (오차)
        ax = axes[3, j]
        diff = torch.abs(ground_truth_sample[j] - mean_pred_sample[j])
        im = ax.imshow(diff, cmap='hot', vmin=0, vmax=CONFIG['DATA_MAX']/2)
        ax.set_title(f'Absolute Error')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"meta_prediction_sample_{sample_idx+1}.png")
    print(f"Saved meta-prediction visualization to meta_prediction_sample_{sample_idx+1}.png")
    plt.show()

def main():
    set_seed(CONFIG['SEED'])
    device = get_device()

    # --- 2. 데이터 준비 ---
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
    
    # DataLoader는 각 태스크를 하나의 배치 아이템으로 다룸 (batch_size=1)
    meta_train_loader = DataLoader(meta_train_dataset, batch_size=1, shuffle=True, num_workers=2)

    # 검증용 태스크를 미리 고정하여 일관된 평가 수행
    val_task_generator = MetaSolarPredictionDataset(base_val_dataset, 1, CONFIG['K_SHOT'], CONFIG['K_QUERY'])
    val_task = val_task_generator[0]
    print(f"Validation task created with {val_task[0].shape[0]} support samples and {val_task[2].shape[0]} query samples.")

    # --- 3. 모델 및 옵티마이저 정의 ---
    # BayesianUNet을 포함하는 MetaLearner를 인스턴스화
    meta_learner = MetaLearner(CONFIG).to(device)
    meta_optimizer = optim.AdamW(meta_learner.parameters(), lr=CONFIG['META_LR'])
    
    # --- 4. 메타-훈련 루프 ---
    best_val_loss = float('inf')
    print("\n--- Starting Meta-Training ---")
    for epoch in range(CONFIG['EPOCHS']):
        
        train_loss = meta_train_one_epoch(meta_learner, meta_train_loader, meta_optimizer, device)
        print(f"\nEpoch {epoch+1}/{CONFIG['EPOCHS']}, Meta-Train Outer Loss: {train_loss:.6f}")
        
        # 5 에포크마다 검증 태스크에 대한 성능 평가
        if (epoch + 1) % 5 == 0:
            val_loss, _, _, _ = meta_evaluate(
                meta_learner, 
                val_task, 
                device, 
                CONFIG['NUM_ADAPTATION_STEPS'], 
                CONFIG['NUM_EVAL_SAMPLES']
            )
            print(f"Validation Task Loss after adaptation: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(meta_learner.state_dict(), CONFIG['MODEL_SAVE_PATH'])
                print(f"Meta-model saved to {CONFIG['MODEL_SAVE_PATH']}")

    print("\n--- Meta-Training Finished ---")

    # --- 5. 최종 평가 및 시각화 ---
    print("\n--- Final Evaluation on a New Test Task ---")
    meta_learner.load_state_dict(torch.load(CONFIG['MODEL_SAVE_PATH']))
    
    # 새로운 테스트 태스크 생성
    test_task_generator = MetaSolarPredictionDataset(base_val_dataset, 1, CONFIG['K_SHOT'], CONFIG['K_QUERY'])
    test_task = test_task_generator[0]
    
    test_loss, mean_pred, std_pred, ground_truth = meta_evaluate(
        meta_learner, 
        test_task, 
        device, 
        CONFIG['NUM_ADAPTATION_STEPS'], 
        CONFIG['NUM_EVAL_SAMPLES']
    )
    print(f"Final Test Task Loss: {test_loss:.6f}")

    # 결과 시각화
    visualize_meta_predictions(mean_pred, std_pred, ground_truth, sample_idx=0)

if __name__ == '__main__':
    main()