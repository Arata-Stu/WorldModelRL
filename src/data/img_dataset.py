import os
import random
import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class H5Dataset(Dataset):
    """
    h5 ファイルから画像データを読み込み、各画像を1サンプルとして返す Dataset

    オプションで、サンプルリスト（(ファイルパス, インデックス) のタプル）を渡すと、
    そのリストに基づいてデータを返します。
    """
    def __init__(self, data_dir, transform=None, samples=None):
        """
        Args:
            data_dir (str): h5 ファイルが保存されているディレクトリ
            transform: torchvision.transforms などの画像前処理
            samples (list or None): (h5_path, index) のタプルリスト。None の場合はディレクトリ内全サンプルを対象にする。
        """
        self.data_dir = data_dir
        self.transform = transform

        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            # 拡張子が .h5 または .hdf5 のファイルを対象とする
            h5_paths = sorted([os.path.join(data_dir, f) 
                               for f in os.listdir(data_dir)
                               if f.endswith('.h5') or f.endswith('.hdf5')])
            for h5_path in h5_paths:
                with h5py.File(h5_path, 'r') as f:
                    n = len(f["observations"])
                    for i in range(n):
                        self.samples.append((h5_path, i))
            print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        h5_path, img_idx = self.samples[idx]
        with h5py.File(h5_path, 'r') as f:
            obs = f["observations"][img_idx]  # (H, W, C)
        # PIL Image に変換して transform を適用
        image = Image.fromarray(obs)
        if self.transform:
            image = self.transform(image)
        return image


class H5DataModule(pl.LightningDataModule):
    """
    h5 ファイルから読み込んだ画像データを用いる DataModule

    全ファイル内のサンプルからシャッフル・分割し、train/validation 用の Dataset を生成します。
    """
    def __init__(self, data_dir, batch_size=32, img_size=64, train_ratio=0.8, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        # ディレクトリ内の全 h5 ファイルからサンプルリストを作成
        samples = []
        h5_files = sorted([os.path.join(self.data_dir, f)
                           for f in os.listdir(self.data_dir)
                           if f.endswith('.h5') or f.endswith('.hdf5')])
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                n = len(f["observations"])
                for i in range(n):
                    samples.append((h5_file, i))
        # サンプル全体をシャッフルして分割
        random.shuffle(samples)
        split_idx = int(len(samples) * self.train_ratio)
        self.train_samples = samples[:split_idx]
        self.val_samples = samples[split_idx:]
        print(f"Train samples: {len(self.train_samples)}")
        print(f"Validation samples: {len(self.val_samples)}")
        
        # 個別のサンプルリストを渡して Dataset を生成
        self.train_dataset = H5Dataset(data_dir=self.data_dir, transform=self.transform, samples=self.train_samples)
        self.val_dataset = H5Dataset(data_dir=self.data_dir, transform=self.transform, samples=self.val_samples)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers)
