# dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import re
from datetime import datetime, timedelta

def _extract_time_from_path(file_path):
    """Extracts datetime from a filepath with format 'YYYY-MM-DD_hhmm.npy'."""
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
            
        file_times = {fp: _extract_time_from_path(fp) for fp in self.file_paths}
        
        for i in range(len(self.file_paths) - (self.input_len + self.target_len) + 1):
            sequence_files = self.file_paths[i : i + self.input_len + self.target_len]
            
            is_continuous = True
            for k in range(len(sequence_files) - 1):
                time1 = file_times.get(sequence_files[k])
                time2 = file_times.get(sequence_files[k+1])
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
        start_index = self.valid_indices[idx]
        
        input_paths = self.file_paths[start_index : start_index + self.input_len]
        target_paths = self.file_paths[start_index + self.input_len : start_index + self.input_len + self.target_len]

        # 입력 시퀀스 로드 및 스택
        input_sequence = [np.load(p) for p in input_paths]
        input_tensor = torch.from_numpy(np.stack(input_sequence, axis=0)).float()

        # 타겟 시퀀스 로드 및 스택
        target_sequence = [np.load(p) for p in target_paths]
        target_tensor = torch.from_numpy(np.stack(target_sequence, axis=0)).float()

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
            
        return input_tensor, target_tensor