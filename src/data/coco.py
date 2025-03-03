import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection

def collate_fn_ignore_target(batch):
    # バッチ内の各サンプルから画像だけを抽出
    images = [sample[0] for sample in batch]
    images = torch.stack(images, dim=0)
    return images

class CocoDataModule(pl.LightningDataModule):
    def __init__(self, base_dir: str, img_size: int=224, batch_size: int=32, num_workers: int=10):
        super().__init__()
        self.train_dir = f'{base_dir}/images/train2017'
        self.train_ann = f'{base_dir}/annotations/instances_train2017.json'
        self.val_dir = f'{base_dir}/images/val2017'
        self.val_ann = f'{base_dir}/annotations/instances_val2017.json'
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        self.train_dataset = CocoDetection(root=self.train_dir, annFile=self.train_ann, transform=self.transform)
        self.val_dataset = CocoDetection(root=self.val_dir, annFile=self.val_ann, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_fn_ignore_target
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn_ignore_target
        )
