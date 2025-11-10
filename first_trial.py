import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import glob
from datetime import datetime, timedelta

# ==============================================================================
# 1. U-Net 모델 구성 요소 (Building Blocks)
# ==============================================================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # <<< --- 여기가 수정된 부분 --- >>>
        # Concatenate 이후의 채널 수를 정확히 계산하여 DoubleConv에 전달합니다.
        # Skip connection 채널(in_channels // 2) + Upsample된 채널(in_channels)
        self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ==============================================================================
# 2. 메인 모델 (AC_UNet_MultiStep)
# ==============================================================================

class AC_UNet_MultiStep(nn.Module):
    def __init__(self, n_channels_in, n_frames_out):
        super(AC_UNet_MultiStep, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_frames_out = n_frames_out

        # Encoder
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_frames_out)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final output
        logits = self.outc(x)
        return logits

# ==============================================================================
# 3. 데이터 로더 (SolarDataset)
# (이 부분은 변경사항 없음)
# ==============================================================================

class SolarDataset(Dataset):
    def __init__(self, data_dir, n_input_frames=5, n_output_frames=4):
        self.data_dir = data_dir
        self.n_input_frames = n_input_frames
        self.n_output_frames = n_output_frames
        self.total_frames_in_sequence = n_input_frames + n_output_frames
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        self.valid_indices = self._get_valid_indices()
        self.min_val = 0.0
        self.max_val = 26.41

    def _get_valid_indices(self):
        valid_indices = []
        for i in range(len(self.files) - self.total_frames_in_sequence + 1):
            is_continuous = True
            for j in range(self.total_frames_in_sequence - 1):
                try:
                    time_current_str = os.path.basename(self.files[i+j]).replace('.npy', '')
                    time_next_str = os.path.basename(self.files[i+j+1]).replace('.npy', '')
                    time_current = datetime.strptime(time_current_str, '%Y-%m-%d_%H%M')
                    time_next = datetime.strptime(time_next_str, '%Y-%m-%d_%H%M')
                    if time_next - time_current != timedelta(minutes=30):
                        is_continuous = False
                        break
                except ValueError:
                    is_continuous = False
                    break
            if is_continuous:
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        input_files = self.files[start_idx : start_idx + self.n_input_frames]
        output_files = self.files[start_idx + self.n_input_frames : start_idx + self.total_frames_in_sequence]
        input_frames = [np.load(f) for f in input_files]
        input_tensor = torch.from_numpy(np.stack(input_frames, axis=0)).float()
        output_frames = [np.load(f) for f in output_files]
        output_tensor = torch.from_numpy(np.stack(output_frames, axis=0)).float()
        input_tensor = 2 * (input_tensor - self.min_val) / (self.max_val - self.min_val) - 1
        output_tensor = 2 * (output_tensor - self.min_val) / (self.max_val - self.min_val) - 1
        return input_tensor, output_tensor

# ==============================================================================
# 4. 학습 및 추론 파이프라인
# (이 부분은 변경사항 없음)
# ==============================================================================

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

def predict(model, input_tensor, device):
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.squeeze(0).cpu()

# ==============================================================================
# 5. 메인 실행 블록
# (이 부분은 변경사항 없음)
# ==============================================================================

if __name__ == '__main__':
    DATA_DIRECTORY = "/home/user/hanwool/new_npy"
    N_INPUT_FRAMES = 5
    N_OUTPUT_FRAMES = 4
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    VAL_SPLIT_RATIO = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    if not os.path.isdir(DATA_DIRECTORY):
        print(f"Error: Data directory not found at '{DATA_DIRECTORY}'")
        print("Please update the DATA_DIRECTORY variable with the correct path.")
    else:
        dataset = SolarDataset(
            data_dir=DATA_DIRECTORY,
            n_input_frames=N_INPUT_FRAMES,
            n_output_frames=N_OUTPUT_FRAMES
        )
        print(f"Total number of valid sequences: {len(dataset)}")
        val_size = int(len(dataset) * VAL_SPLIT_RATIO)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")

        model = AC_UNet_MultiStep(n_channels_in=N_INPUT_FRAMES, n_frames_out=N_OUTPUT_FRAMES)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print("\nStarting model training...")
        train_model(model, train_loader, val_loader, optimizer, criterion, EPOCHS, device)
        print("Training finished.")
        
        print("\nRunning inference example...")
        sample_input, sample_target = val_dataset[0]
        prediction = predict(model, sample_input, device)
        print(f"Input tensor shape: {sample_input.shape}")
        print(f"Target tensor shape: {sample_target.shape}")
        print(f"Prediction tensor shape: {prediction.shape}")
        sample_mse = criterion(prediction, sample_target.to('cpu'))
        print(f"MSE on single sample prediction: {sample_mse.item():.4f}")