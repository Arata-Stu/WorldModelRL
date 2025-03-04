from omegaconf import DictConfig

from src.data.coco import CocoDataModule
from src.data.img_dataset import H5DataModule
from src.data.mdn_dataset import WorldModelDataModule

def get_data_module(data_cfg: DictConfig):
    
    if data_cfg.name == "coco":
        print(f"Loading data module for {data_cfg.name} dataset")
        return CocoDataModule(base_dir=data_cfg.data_dir,
                              img_size=data_cfg.img_size,
                              batch_size=data_cfg.batch_size,
                              num_workers=data_cfg.num_workers)
    elif data_cfg.name == "gym_img":
        print(f"Loading data module for {data_cfg.name} dataset")
        return H5DataModule(data_dir=data_cfg.data_dir,
                          batch_size=data_cfg.batch_size,
                          img_size=data_cfg.img_size,
                          train_ratio=data_cfg.train_ratio,
                          num_workers=data_cfg.num_workers)
    elif data_cfg.name == "mdn":
        print(f"Loading data module for {data_cfg.name} dataset")
        return WorldModelDataModule(data_dir=data_cfg.data_dir,
                                batch_size=data_cfg.batch_size,
                                seq_length=data_cfg.seq_length,
                                num_workers=data_cfg.num_workers)
    else:
        NotImplementedError(f"Data module for {data_cfg.name} dataset is not implemented")