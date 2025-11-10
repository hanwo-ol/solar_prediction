# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from datetime import datetime
from torch.cuda.amp import GradScaler

# 모듈 임포트
from model import UNetMultiStep
from dataset import SolarPredictionDataset, _extract_time_from_path
from engine import train_one_epoch, evaluate
from utils import set_seed, get_device

# --- 1. 설정 (Configuration) ---
CONFIG = {
    "DATA_DIR": "/home/user/hanwool/new_npy",
    "MODEL_SAVE_PATH": "./best_multistep_model_4to4.pth",
    "SEED": 42,
    "BATCH_SIZE": 8,
    "EPOCHS": 20, # 에포크 수 증가
    "LEARNING_RATE": 1e-4,
    "NUM_WORKERS": 4,
    "INPUT_LEN": 4,      # 입력 시퀀스 길이 (과거 4개 프레임, 2시간)
    "TARGET_LEN": 4,     # 출력 시퀀스 길이 (미래 4개 프레임, 2시간)
    "DATA_MIN": 0.0,     # 데이터 정규화를 위한 최소값
    "DATA_MAX": 26.41    # 데이터 정규화를 위한 최대값
}

def visualize_predictions(model, dataloader, device, num_samples=3):
    """테스트 데이터셋으로 예측하고 결과를 시각화합니다."""
    model.eval()
    print("\n--- Visualizing Predictions ---")
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # -1~1 범위를 원래 데이터 범위로 되돌리는 함수
            def denormalize(tensor):
                val_range = CONFIG['DATA_MAX'] - CONFIG['DATA_MIN']
                return (tensor.cpu() * (val_range / 2.0)) + ((CONFIG['DATA_MAX'] + CONFIG['DATA_MIN']) / 2.0)

            targets_denorm = denormalize(targets)
            outputs_denorm = denormalize(outputs)

            # 시각화를 위해 첫 번째 배치 아이템만 사용
            targs = targets_denorm[0]
            preds = outputs_denorm[0]

            fig, axes = plt.subplots(3, CONFIG['TARGET_LEN'], figsize=(CONFIG['TARGET_LEN'] * 4, 10))
            fig.suptitle(f'Sample {i+1}: Prediction vs. Ground Truth\n(Predicting t+30m to t+120m)', fontsize=16)

            for j in range(CONFIG['TARGET_LEN']):
                time_step = 30 * (j + 1)
                
                # 1행: Ground Truth
                ax = axes[0, j]
                im = ax.imshow(targs[j], cmap='gray', vmin=CONFIG['DATA_MIN'], vmax=CONFIG['DATA_MAX'])
                ax.set_title(f'Target (t+{time_step}m)')
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 2행: Prediction
                ax = axes[1, j]
                im = ax.imshow(preds[j], cmap='gray', vmin=CONFIG['DATA_MIN'], vmax=CONFIG['DATA_MAX'])
                ax.set_title(f'Prediction (t+{time_step}m)')
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 3행: Difference (오차)
                ax = axes[2, j]
                diff = torch.abs(targs[j] - preds[j])
                im = ax.imshow(diff, cmap='hot', vmin=0, vmax=CONFIG['DATA_MAX']/2)
                ax.set_title(f'Difference (t+{time_step}m)')
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"prediction_sample_{i+1}.png")
            print(f"Saved prediction visualization to prediction_sample_{i+1}.png")
            plt.show()


def main():
    set_seed(CONFIG['SEED'])
    device = get_device()

    # --- 2. 데이터 준비 ---
    data_dir = Path(CONFIG['DATA_DIR'])
    all_files = sorted(list(data_dir.glob("*.npy")))
    if not all_files:
        raise FileNotFoundError(f"Error: No .npy files found in {data_dir}. Please check the path.")
        
    # 연도 기반 데이터 분할 (2021, 2022 -> train, 2023 -> val/test)
    train_files = [p for p in all_files if _extract_time_from_path(p) and _extract_time_from_path(p).year in [2021, 2022]]
    val_test_files = [p for p in all_files if _extract_time_from_path(p) and _extract_time_from_path(p).year == 2023]
    
    val_split_index = int(len(val_test_files) * 0.8)
    val_files = val_test_files[:val_split_index]
    test_files = val_test_files[val_split_index:]

    print(f"Total files: {len(all_files)}")
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

    # -1과 1 사이로 정규화하는 변환 함수
    transform = transforms.Lambda(
        lambda x: (x - (CONFIG['DATA_MAX'] + CONFIG['DATA_MIN']) / 2.0) / ((CONFIG['DATA_MAX'] - CONFIG['DATA_MIN']) / 2.0)
    )

    train_dataset = SolarPredictionDataset(train_files, CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'], transform)
    val_dataset = SolarPredictionDataset(val_files, CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'], transform)
    test_dataset = SolarPredictionDataset(test_files, CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'], transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)

    # --- 3. 모델, 손실 함수, 옵티마이저, AMP 스케일러 정의 ---
    model = UNetMultiStep(
        n_channels=CONFIG['INPUT_LEN'], 
        num_future_steps=CONFIG['TARGET_LEN']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    scaler = GradScaler() # AMP 스케일러 초기화

    # --- 4. 훈련 및 검증 루프 ---
    best_val_loss = float('inf')
    print("\n--- Starting Training ---")
    for epoch in range(CONFIG['EPOCHS']):
        print(f"\nEpoch {epoch+1}/{CONFIG['EPOCHS']}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr > new_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['MODEL_SAVE_PATH'])
            print(f"Model saved to {CONFIG['MODEL_SAVE_PATH']}")

    print("\n--- Training Finished ---")

    # --- 5. 테스트 및 시각화 ---
    print("\n--- Running Inference on Test Set ---")
    model.load_state_dict(torch.load(CONFIG['MODEL_SAVE_PATH']))
    
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.6f}")

    visualize_predictions(model, test_loader, device, num_samples=3)

if __name__ == '__main__':
    main()