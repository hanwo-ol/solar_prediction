# engine.py
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # AMP: autocast 컨텍스트 내에서 forward pass 실행
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # AMP: scaler를 사용하여 loss의 스케일을 조정하고 backward pass 실행
        scaler.scale(loss).backward()
        
        # AMP: scaler를 사용하여 optimizer.step() 실행 (unscaling 포함)
        scaler.step(optimizer)
        
        # AMP: 다음 반복을 위해 scaler 업데이트
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 평가 시에도 autocast를 사용하여 속도 향상 가능
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
    return total_loss / len(dataloader)