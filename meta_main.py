# meta_main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import math

# 모듈 임포트
from meta_model import MetaLearner
from meta_dataset import SolarPredictionDataset, MetaSolarPredictionDataset, _extract_time_from_path
from meta_engine import meta_train_one_epoch, meta_evaluate
from utils import set_seed, get_device

# --- 1. 설정 (Configuration) ---
CONFIG = {
    "DATA_DIR": "/home/user/hanwool/new_npy",
    "MODEL_SAVE_PATH": "./best_bayesian_meta_model_seasonal_split.pth",
    "SEED": 42,
    "EPOCHS": 50,
    "META_LR": 1e-7,
    "INNER_LR": 1e-6,
    "INNER_STEPS": 5,
    "KL_WEIGHT_INIT": 1e-8,
    "KL_WEIGHT_MAX": 1e-6,
    "GRAD_CLIP_NORM": 5.0,
    "TASKS_PER_EPOCH": 20,
    "K_SHOT": 5,
    "K_QUERY": 10,
    "NUM_ADAPTATION_STEPS": 10,
    "NUM_EVAL_SAMPLES": 20,
    "INPUT_LEN": 4,
    "TARGET_LEN": 4,
    "DATA_MIN": 0.0,
    "DATA_MAX": 26.41
}

def visualize_meta_predictions(mean_pred, std_pred, ground_truth, sample_idx=0):
    """메타-러닝 평가 결과를 시각화하는 함수."""
    print("\n--- Visualizing Final Test Task Prediction ---")
    
    def denormalize(tensor):
        val_range = CONFIG['DATA_MAX'] - CONFIG['DATA_MIN']
        return (tensor.cpu().numpy() * (val_range / 2.0)) + ((CONFIG['DATA_MAX'] + CONFIG['DATA_MIN']) / 2.0)

    gt_sequence = denormalize(ground_truth[sample_idx])
    mean_sequence = denormalize(mean_pred[sample_idx])
    std_sequence = std_pred[sample_idx].cpu().numpy()

    num_steps = CONFIG['TARGET_LEN']
    fig, axes = plt.subplots(3, num_steps, figsize=(num_steps * 4, 10))
    fig.suptitle(f'Test Task Adaptation Result (Sample {sample_idx+1})\nPrediction, Uncertainty, and Ground Truth', fontsize=16)

    for j in range(num_steps):
        time_step = 30 * (j + 1)
        
        ax = axes[0, j]
        im = ax.imshow(gt_sequence[j], cmap='gray', vmin=CONFIG['DATA_MIN'], vmax=CONFIG['DATA_MAX'])
        ax.set_title(f'Target (t+{time_step}m)')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, j]
        im = ax.imshow(mean_sequence[j], cmap='gray', vmin=CONFIG['DATA_MIN'], vmax=CONFIG['DATA_MAX'])
        ax.set_title(f'Mean Prediction (t+{time_step}m)')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[2, j]
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

    # --- 2. 데이터 준비 (수정된 분할 전략) ---
    data_dir = Path(CONFIG['DATA_DIR'])
    all_files = sorted(list(data_dir.glob("*.npy")))
    if not all_files:
        raise FileNotFoundError(f"Error: No .npy files found in {data_dir}.")

    # --- [핵심 수정] 계절적 특성을 고려한 데이터 분할 ---
    TRAIN_MONTHS = {1, 3, 4, 6, 7, 9, 10, 12}
    VAL_MONTHS = {2, 5, 8, 11}

    train_files = []
    val_files = []
    test_files = []

    print("Splitting data based on seasonal strategy...")
    for p in all_files:
        dt = _extract_time_from_path(p)
        if dt is None:
            continue

        if dt.year in [2021, 2022]:
            if dt.month in TRAIN_MONTHS:
                train_files.append(p)
            elif dt.month in VAL_MONTHS:
                val_files.append(p)
        elif dt.year == 2023:
            test_files.append(p)
    # ----------------------------------------------------

    print(f"Total files: {len(all_files)}")
    print(f"Train files: {len(train_files)} (Years 2021-2022, Months: {sorted(list(TRAIN_MONTHS))})")
    print(f"Val files: {len(val_files)} (Years 2021-2022, Months: {sorted(list(VAL_MONTHS))})")
    print(f"Test files: {len(test_files)} (Year 2023)")

    transform = transforms.Lambda(
        lambda x: (x - (CONFIG['DATA_MAX'] + CONFIG['DATA_MIN']) / 2.0) / ((CONFIG['DATA_MAX'] - CONFIG['DATA_MIN']) / 2.0)
    )

    # 각 분할에 대한 기본 데이터셋 생성
    base_train_dataset = SolarPredictionDataset(train_files, CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'], transform)
    base_val_dataset = SolarPredictionDataset(val_files, CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'], transform)
    base_test_dataset = SolarPredictionDataset(test_files, CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'], transform)

    # 메타-러닝용 데이터셋(태스크 생성기) 생성
    meta_train_dataset = MetaSolarPredictionDataset(base_train_dataset, CONFIG['TASKS_PER_EPOCH'], CONFIG['K_SHOT'], CONFIG['K_QUERY'])
    meta_train_loader = DataLoader(meta_train_dataset, batch_size=1, shuffle=True, num_workers=2)

    # 검증 및 테스트를 위한 고정된 단일 태스크 생성
    val_task_generator = MetaSolarPredictionDataset(base_val_dataset, 1, CONFIG['K_SHOT'], CONFIG['K_QUERY'])
    val_task = val_task_generator[0]
    print(f"Validation task created from unseen months with {val_task[0].shape[0]} support and {val_task[2].shape[0]} query samples.")

    test_task_generator = MetaSolarPredictionDataset(base_test_dataset, 1, CONFIG['K_SHOT'], CONFIG['K_QUERY'])
    test_task = test_task_generator[0]
    print(f"Test task created from unseen year (2023) with {test_task[0].shape[0]} support and {test_task[2].shape[0]} query samples.")

    # --- 3. 모델, 옵티마이저, 스케줄러 정의 ---
    meta_learner = MetaLearner(CONFIG).to(device)
    meta_optimizer = optim.AdamW(meta_learner.parameters(), lr=CONFIG['META_LR'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, 'min', patience=5, factor=0.5)

    # --- 4. 메타-훈련 루프 ---
    best_val_loss = float('inf')
    print("\n--- Starting Meta-Training ---")
    for epoch in range(CONFIG['EPOCHS']):
        kl_weight = min(CONFIG['KL_WEIGHT_MAX'], CONFIG['KL_WEIGHT_INIT'] + (CONFIG['KL_WEIGHT_MAX'] - CONFIG['KL_WEIGHT_INIT']) * (2 * epoch / CONFIG['EPOCHS']))
        meta_learner.config['KL_WEIGHT'] = kl_weight
        
        train_loss = meta_train_one_epoch(meta_learner, meta_train_loader, meta_optimizer, device, CONFIG['GRAD_CLIP_NORM'])
        
        val_loss, _, _, _ = meta_evaluate(meta_learner, val_task, device, CONFIG['NUM_ADAPTATION_STEPS'], CONFIG['NUM_EVAL_SAMPLES'])
        
        print(f"\nEpoch {epoch+1}/{CONFIG['EPOCHS']} | KL Weight: {kl_weight:.2e} | Meta-Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if not math.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(meta_learner.state_dict(), CONFIG['MODEL_SAVE_PATH'])
            print(f"Best model saved to {CONFIG['MODEL_SAVE_PATH']} with validation loss: {best_val_loss:.6f}")

        if not math.isnan(val_loss):
            scheduler.step(val_loss)

    print("\n--- Meta-Training Finished ---")

    # --- 5. 최종 평가 및 시각화 ---
    print("\n--- Final Evaluation on a New Test Task from 2023 ---")
    
    if Path(CONFIG['MODEL_SAVE_PATH']).exists():
        meta_learner.load_state_dict(torch.load(CONFIG['MODEL_SAVE_PATH']))
        print("Loaded best model for final evaluation.")
        
        test_loss, mean_pred, std_pred, ground_truth = meta_evaluate(
            meta_learner, 
            test_task, # 이제 명확히 분리된 테스트 태스크를 사용
            device, 
            CONFIG['NUM_ADAPTATION_STEPS'], 
            CONFIG['NUM_EVAL_SAMPLES']
        )
        
        if not math.isnan(test_loss) and mean_pred is not None:
            print(f"Final Test Task Loss: {test_loss:.6f}")
            visualize_meta_predictions(mean_pred, std_pred, ground_truth, sample_idx=0)
        else:
            print("Final evaluation failed: NaN was produced during adaptation.")
    else:
        print("Could not find a saved model. Final evaluation skipped.")

if __name__ == '__main__':
    main()