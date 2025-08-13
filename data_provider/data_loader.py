import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# === Stub classes for unused datasets (can be removed if not needed) ===
class Dataset_ETT_hour:
    pass
class Dataset_ETT_minute:
    pass
class Dataset_Custom:
    pass
class Dataset_Solar:
    pass
class Dataset_PEMS:
    pass
class Dataset_Pred:
    pass


class BPRegressionDataset(Dataset):
    def __init__(self, data_dir, split='train', size=None, scale=True, timeenc=0, freq='h', target=None):
        self.data_dir = data_dir
        self.split = split
        self.scale = scale
        self.size = size if size else [1000, 64, 0]  #1000 PPG samples in each segment
        self.seq_len = self.size[0]

        # 1. Load the complete dataset first
        ppg_path = os.path.join(data_dir, 'ppg_train.csv')
        label_path = os.path.join(data_dir, 'bp_labels_train.csv')

        try:
            ppg_full = pd.read_csv(ppg_path).values.astype(np.float32)
            bp_full = pd.read_csv(label_path).values.astype(np.float32)
        except FileNotFoundError as e:
            print(f"Error: {e}. Make sure you have run the data cleaning script first.")
            raise e

        # 2. Define split boundaries (70% train, 10% validation, 20% test)
        n_samples = len(ppg_full)
        train_end_idx = int(n_samples * 0.7)
        val_end_idx = int(n_samples * 0.8)

        if split == 'train':
            start_idx, end_idx = 0, train_end_idx
        elif split == 'val':
            start_idx, end_idx = train_end_idx, val_end_idx
        elif split == 'test':
            start_idx, end_idx = val_end_idx, n_samples
        else:
            raise ValueError(f"Invalid split: '{split}'. Must be 'train', 'val', or 'test'.")

        # 3. Handle PPG scaling
        if self.scale:
            self.input_scaler = StandardScaler()
            train_data_for_scaling = ppg_full[0:train_end_idx]
            self.input_scaler.fit(train_data_for_scaling)
            ppg_full_scaled = self.input_scaler.transform(ppg_full)
            self.ppg_data = ppg_full_scaled[start_idx:end_idx]
        else:
            self.input_scaler = None
            self.ppg_data = ppg_full[start_idx:end_idx]

        # 4. Handle BP label scaling
        if self.scale:
            self.label_scaler = StandardScaler()
            self.label_scaler.fit(bp_full[0:train_end_idx])
            bp_scaled = self.label_scaler.transform(bp_full)
            self.bp_labels = bp_scaled[start_idx:end_idx]
        else:
            self.label_scaler = None
            self.bp_labels = bp_full[start_idx:end_idx]

    def __len__(self):
        return len(self.ppg_data)

    def __getitem__(self, idx):
        full_seq = self.ppg_data[idx]
        target = self.bp_labels[idx]

        # Crop or pad PPG signal
        if len(full_seq) >= self.seq_len:
            seq = full_seq[:self.seq_len]
        else:
            pad = np.zeros(self.seq_len, dtype=np.float32)
            pad[:len(full_seq)] = full_seq
            seq = pad

        seq_x = torch.tensor(seq, dtype=torch.float32).unsqueeze(-1)
        seq_y = torch.tensor(target, dtype=torch.float32)
        seq_x_mark = torch.zeros_like(seq_x)
        seq_y_mark = torch.zeros((1, 1), dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def inverse_transform(self, data):
        """
        Inverse transform for predicted label values (BP).
        Expects shape [batch_size, 2]
        """
        if self.label_scaler is not None:
            return self.label_scaler.inverse_transform(data)
        return data
