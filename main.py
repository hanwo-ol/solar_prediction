# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

# 모듈 임포트
from model import UNetMultiStep
from dataset import SolarPredictionDataset
from engine import train_one_epoch, evaluate
from utils import set_seed, get_device

# --- 1. 설정 (Configuration) ---
CONFIG = {
    "DATA_DIR": "/home/user/hanwool/new_npy", # 실제 데이터 경로
    "MODEL_SAVE_PATH": "./best_multistep_model.pth",
    "SEED": 42,
    "BATCH_SIZE": 8,
    "EPOCHS": 10, # 실제 학습 시에는 더 길게 설정 (예: 50)
    "LEARNING_RATE": 1e-4,
    "NUM_WORKERS": 4,
    "INPUT_LEN": 5,      # 과거 5개 프레임 입력
    "TARGET_LEN": 4,     # 미래 4개 프레임(30, 60, 90, 120분) 예측
    "IMG_HEIGHT": 512,   # 이미지 높이
    "IMG_WIDTH": 512,    # 이미지 너비
    # 데이터 정규화를 위한 값 (훈련 데이터셋에서 미리 계산해야 함)
    "DATA_MIN": 0.0,
    "DATA_MAX": 26.4 # 수정 필요
}

def create_dummy_data(base_dir, num_files=100):
    """테스트를 위한 더미 npy 파일 생성"""
    print(f"Creating dummy data in {base_dir}...")
    p = Path(base_dir)
    p.mkdir(exist_ok=True)
    start_time = datetime(2023, 1, 1, 0, 0)
    for i in range(num_files):
        timestamp = start_time + timedelta(minutes=30 * i)
        filename = timestamp.strftime("%Y-%m-%d_%H%M.npy")
        data = np.random.rand(CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH']).astype(np.float32) * CONFIG['DATA_MAX']
        np.save(p / filename, data)
    print("Dummy data created.")

def visualize_predictions(model, dataloader, device, num_samples=1):
    """테스트 데이터셋으로 예측하고 결과를 시각화"""
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 정규화 해제
            inputs_denorm = (inputs.cpu() * (CONFIG['DATA_MAX'] - CONFIG['DATA_MIN']) / 2.0) + (CONFIG['DATA_MAX'] + CONFIG['DATA_MIN']) / 2.0
            targets_denorm = (targets.cpu() * (CONFIG['DATA_MAX'] - CONFIG['DATA_MIN']) / 2.0) + (CONFIG['DATA_MAX'] + CONFIG['DATA_MIN']) / 2.0
            outputs_denorm = (outputs.cpu() * (CONFIG['DATA_MAX'] - CONFIG['DATA_MIN']) / 2.0) + (CONFIG['DATA_MAX'] + CONFIG['DATA_MIN']) / 2.0

            # 시각화를 위해 배치 중 첫 번째 샘플 선택
            inp_last = inputs_denorm[0, -1, :, :] # 입력의 마지막 프레임
            targs = targets_denorm[0] # (4, H, W)
            preds = outputs_denorm[0] # (4, H, W)

            fig, axes = plt.subplots(3, CONFIG['TARGET_LEN'], figsize=(15, 10))
            
            # 제목 설정
            axes[0, 0].set_title("Input (t)")
            for j in range(CONFIG['TARGET_LEN']):
                axes[0, j+1 if j < CONFIG['TARGET_LEN']-1 else 0].set_title(f"Target (t+{30*(j+1)}m)")
                axes[1, j].set_title(f"Prediction (t+{30*(j+1)}m)")
                axes[2, j].set_title(f"Difference (t+{30*(j+1)}m)")

            # 이미지 표시
            im_in = axes[0, 0].imshow(inp_last, cmap='gray')
            fig.colorbar(im_in, ax=axes[0,0])

            for j in range(CONFIG['TARGET_LEN']):
                im_t = axes[0, j].imshow(targs[j], cmap='gray')
                im_p = axes[1, j].imshow(preds[j], cmap='gray')
                diff = torch.abs(targs[j] - preds[j])
                im_d = axes[2, j].imshow(diff, cmap='hot')

                fig.colorbar(im_t, ax=axes[0,j])
                fig.colorbar(im_p, ax=axes[1,j])
                fig.colorbar(im_d, ax=axes[2,j])

            for ax_row in axes:
                for ax in ax_row:
                    ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"prediction_sample_{i}.png")
            plt.show()


def main():
    set_seed(CONFIG['SEED'])
    device = get_device()

    # --- 2. 데이터 준비 ---
    data_dir = Path(CONFIG['DATA_DIR'])
    # # 만약 더미 데이터가 필요하다면 아래 주석을 해제하세요.
    # if not any(data_dir.iterdir()):
    #      create_dummy_data(data_dir)

    all_files = list(data_dir.glob("*.npy"))
    if not all_files:
        print(f"Error: No .npy files found in {data_dir}. Please check the path.")
        return
        
    # 데이터 분할 (예: 80% train, 10% val, 10% test)
    train_size = int(0.8 * len(all_files))
    val_size = int(0.1 * len(all_files))
    train_files = all_files[:train_size]
    val_files = all_files[train_size : train_size + val_size]
    test_files = all_files[train_size + val_size :]

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

    # 정규화 변환 정의
    transform = transforms.Lambda(
        lambda x: (x - (CONFIG['DATA_MAX'] + CONFIG['DATA_MIN']) / 2.0) / ((CONFIG['DATA_MAX'] - CONFIG['DATA_MIN']) / 2.0)
    )

    train_dataset = SolarPredictionDataset(train_files, CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'], transform)
    val_dataset = SolarPredictionDataset(val_files, CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'], transform)
    test_dataset = SolarPredictionDataset(test_files, CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'], transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'])

    # --- 3. 모델, 손실 함수, 옵티마이저 정의 ---
    model = UNetMultiStep(
        n_channels=CONFIG['INPUT_LEN'], 
        num_future_steps=CONFIG['TARGET_LEN']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])

    # --- 4. 훈련 및 검증 루프 ---
    best_val_loss = float('inf')
    print("\n--- Starting Training ---")
    for epoch in range(CONFIG['EPOCHS']):
        print(f"\nEpoch {epoch+1}/{CONFIG['EPOCHS']}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['MODEL_SAVE_PATH'])
            print(f"Model saved to {CONFIG['MODEL_SAVE_PATH']}")

    print("\n--- Training Finished ---")

    # --- 5. 테스트 및 시각화 ---
    print("\n--- Running Inference on Test Set ---")
    model.load_state_dict(torch.load(CONFIG['MODEL_SAVE_PATH']))
    visualize_predictions(model, test_loader, device, num_samples=3)

if __name__ == '__main__':
    main()