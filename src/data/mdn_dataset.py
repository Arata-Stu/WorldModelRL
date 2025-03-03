import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class MDNRNNDataset(Dataset):
    """
    MDN-RNN の学習データセット。
    各エピソードの `latent` と `actions` を時系列データとして処理する。
    """
    def __init__(self, data_dir, seq_length=20):
        """
        Args:
            data_dir (str): h5 ファイルが格納されたディレクトリ
            seq_length (int): RNN に入力する時系列の長さ
        """
        self.data_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")])
        self.seq_length = seq_length
        self.sequences = []

        # 各 h5 ファイルをロードしてデータを収集
        for file_path in self.data_paths:
            with h5py.File(file_path, 'r') as f:
                if "latent" not in f or "actions" not in f:
                    print(f"Skipping {file_path}, missing latent or actions data.")
                    continue
                
                latent = f["latent"][()]  # (steps, latent_dim)
                actions = f["actions"][()]  # (steps, action_dim)

            # RNN の学習に適した時系列のデータを作成
            num_sequences = len(latent) - seq_length
            for i in range(num_sequences):
                latent_seq = latent[i:i + seq_length]
                action_seq = actions[i:i + seq_length]
                target_z = latent[i + 1:i + 1 + seq_length]  # 次の潜在状態

                self.sequences.append((latent_seq, action_seq, target_z))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        latent_seq, action_seq, target_z = self.sequences[idx]
        return (
            torch.tensor(latent_seq, dtype=torch.float32),
            torch.tensor(action_seq, dtype=torch.float32),
            torch.tensor(target_z, dtype=torch.float32)
        )

class MDNRNNDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, seq_length=20, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = MDNRNNDataset(self.data_dir, self.seq_length)
        
        if len(dataset) == 0:
            raise ValueError("No valid h5 datasets found in the specified directory.")

        # 80% を学習用、20% をバリデーション用に分割
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        print(f"Train: {train_size}, Val: {val_size}")
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)