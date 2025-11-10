# dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import re
from datetime import datetime, timedelta

def _extract_time_from_path(file_path):
    """'YYYY-MM-DD_hhmm.npy' 형식의 파일 경로에서 datetime 객체를 추출합니다."""
    match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{4})', str(file_path.name))
    if not match:
        return None
    date_str, time_str = match.groups()
    try:
        return datetime.strptime(f"{date_str}{time_str}", "%Y-%m-%d%H%M")
    except ValueError:
        return None

class SolarPredictionDataset(Dataset):
    def __init__(self, file_paths, input_len, target_len, transform=None):
        self.file_paths = sorted(file_paths)
        self.input_len = input_len
        self.target_len = target_len
        self.transform = transform
        
        # 연속적인 시퀀스를 구성할 수 있는 유효한 시작 인덱스만 미리 계산
        self.valid_indices = self._find_valid_indices()

    def _find_valid_indices(self):
        valid_starts = []
        if not self.file_paths:
            return valid_starts
            
        # 시간 추출을 한 번만 수행하여 효율성 증대
        file_times = {fp: _extract_time_from_path(fp) for fp in self.file_paths}
        
        total_seq_len = self.input_len + self.target_len
        for i in range(len(self.file_paths) - total_seq_len + 1):
            sequence_files = self.file_paths[i : i + total_seq_len]
            
            is_continuous = True
            for k in range(len(sequence_files) - 1):
                time1 = file_times.get(sequence_files[k])
                time2 = file_times.get(sequence_files[k+1])
                # 시간 정보가 없거나 30분 간격이 아니면 유효하지 않은 시퀀스
                if not time1 or not time2 or (time2 - time1) != timedelta(minutes=30):
                    is_continuous = False
                    break
            
            if is_continuous:
                valid_starts.append(i)
                
        print(f"Found {len(valid_starts)} valid sequences.")
        return valid_starts

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 미리 계산된 유효한 시작 인덱스를 사용
        start_index = self.valid_indices[idx]
        
        input_paths = self.file_paths[start_index : start_index + self.input_len]
        target_paths = self.file_paths[start_index + self.input_len : start_index + self.input_len + self.target_len]

        # 입력 시퀀스 로드 및 스택 (C, H, W)
        input_sequence = [np.load(p)[np.newaxis, :, :] for p in input_paths]
        input_tensor = torch.from_numpy(np.concatenate(input_sequence, axis=0)).float()

        # 타겟 시퀀스 로드 및 스택 (C, H, W)
        target_sequence = [np.load(p)[np.newaxis, :, :] for p in target_paths]
        target_tensor = torch.from_numpy(np.concatenate(target_sequence, axis=0)).float()

        if self.transform:
            # 참고: transform은 각 이미지에 개별 적용되지 않고 텐서 전체에 적용됩니다.
            # 필요시 transform 로직 수정이 필요할 수 있습니다.
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
            
        return input_tensor, target_tensor