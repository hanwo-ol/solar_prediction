# meta_main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import math # isnan 확인을 위해 추가

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
    "EPOCHS": 5,
    "META_LR": 1e-5, # [조정] nan 발생 시 학습률을 낮추는 것이 좋음
    "INNER_LR": 1e-3, # [조정] 내부 루프 학습률도 낮춰서 안정성 확보
    "INNER_STEPS": 5,
    "KL_WEIGHT": 1e-4, # [조정] KL 가중치를 낮춰 초반에 MSE에 집중
    "GRAD_CLIP_NORM": 1.0, # [신규] 그래디언트 클리핑 임계값
    
    # --- 태스크 구성 하이퍼파라미터 ---
    "TASKS_PER_EPOCH": 10,
    "K_SHOT": 5,
    "K_QUERY": 10,
    
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
    # ... (이전 답변의 시각화 코드) ...
    pass # 간결성을 위해 생략

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
        print(f"Final Test Task Loss: {test_loss:.6f}")

        if mean_pred is not None:
            visualize_meta_predictions(mean_pred, std_pred, ground_truth, sample_idx=0)
    else:
        print("Could not find a saved model. Final evaluation skipped.")

if __name__ == '__main__':
    main()