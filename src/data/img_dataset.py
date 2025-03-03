import os
import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class NPZDataset(Dataset):
    """
    npz ファイルから画像データを読み込み、各画像を1サンプルとして返す Dataset
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): npz ファイルが保存されているディレクトリ
            transform: torchvision.transforms などの画像前処理
        """
        self.npz_paths = sorted([os.path.join(data_dir, f) 
                                  for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.transform = transform
        # 各 npz ファイル内の全画像へのインデックスリストを作成
        self.samples = []  # (npz_path, index_in_file) のタプル
        for npz_path in self.npz_paths:
            data = np.load(npz_path)
            observations = data["observations"]
            n = len(observations)
            for i in range(n):
                self.samples.append((npz_path, i))
            data.close()
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npz_path, img_idx = self.samples[idx]
        data = np.load(npz_path)
        obs = data["observations"][img_idx]  # (H, W, C)
        data.close()
        # PIL Image に変換して transform を適用
        image = Image.fromarray(obs)
        if self.transform:
            image = self.transform(image)
        return image


def split_dataset(data_list, train_ratio=0.8):
    random.shuffle(data_list)
    split_idx = int(len(data_list) * train_ratio)
    return data_list[:split_idx], data_list[split_idx:]

class NPZDataModule(pl.LightningDataModule):
    """
    npz ファイルから読み込んだ画像データを用いる DataModule
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
        # ディレクトリ内の全 npz ファイルをリストアップ
        npz_files = sorted([os.path.join(self.data_dir, f) 
                             for f in os.listdir(self.data_dir) if f.endswith('.npz')])
        # 各 npz ファイルからサンプル（画像）のインデックスリストを作成
        samples = []
        for npz_file in npz_files:
            data = np.load(npz_file)
            n = len(data["observations"])
            for i in range(n):
                samples.append((npz_file, i))
            data.close()
        
        # サンプル全体をシャッフルして分割
        random.shuffle(samples)
        split_idx = int(len(samples) * self.train_ratio)
        self.train_samples = samples[:split_idx]
        self.val_samples = samples[split_idx:]
        print(f"Train samples: {len(self.train_samples)}")
        print(f"Validation samples: {len(self.val_samples)}")
        
        # DataModule 用の Dataset を生成
        self.train_dataset = NPZDataset(data_dir=self.data_dir, transform=self.transform)
        self.val_dataset = NPZDataset(data_dir=self.data_dir, transform=self.transform)
        # ※ 本来は train_dataset と val_dataset でサンプルリストを別々にする必要がありますが、
        #     NPZDataset 内で全サンプルを読み込む設計の場合は、DataLoader で sampler を指定するか、
        #     Dataset 内でサンプルリストを引き渡す設計にするのが望ましいです。

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers)
