from functools import partial
import lightning as pl
import torch
import torchvision.utils as vutils
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.dataset import get_data_module
from src.models.VAE.VAE import get_vae
from src.utils.helppers import mask_patches

import torch.nn.functional as F

class VAETrainer(pl.LightningModule):
    def __init__(self, vae_cfg: DictConfig,
                 learning_rate: float,
                 mask_enabled: bool,
                 mask_ratio: float = 0.75,
                 patch_size: int = 16):
        super(VAETrainer, self).__init__()
        self.model = get_vae(vae_cfg=vae_cfg)
        self.learning_rate = learning_rate
        self.mask_enabled = mask_enabled
        self.mask_patches = partial(mask_patches, mask_ratio=mask_ratio, patch_size=patch_size)

        print("mask_enabled:", mask_enabled)
        if mask_enabled:    
            print("mask_ratio:", mask_ratio)
            print("patch_size:", patch_size)
        
    def masked_loss_function(self, recon_x, x, mask, mu, log_var):
        """
        マスクされた部分のみの再構成損失と、通常の KL損失を計算
        ※ (1 - mask) がマスク部分を表す（mask: 非マスク部分が1, マスク部分が0）
        """
        rec_loss = F.mse_loss(recon_x * (1 - mask), x * (1 - mask), reduction='sum')
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()) 
        return rec_loss + kl_loss, rec_loss, kl_loss

    def training_step(self, batch, batch_idx):
        x = batch
        if self.mask_enabled:
            # マスク学習の場合
            masked_x, mask = self.mask_patches(x)
            recon_x, mu, log_var = self.model(masked_x)
            loss, rec_loss, kl_loss = self.masked_loss_function(recon_x, x, mask, mu, log_var)
        else:
            # マスクなしの場合、通常の VAE として全画像を入力
            recon_x, mu, log_var = self.model(x)
            loss, rec_loss, kl_loss = self.model.vae_loss(recon_x, x, mu, log_var)
            
        self.log("train_loss", loss, prog_bar=True)
        self.log("rec_loss", rec_loss)
        self.log("kl_loss", kl_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        if self.mask_enabled:
            masked_x, mask = self.mask_patches(x)
            recon_x, mu, log_var = self.model(masked_x)
            loss, rec_loss, kl_loss = self.masked_loss_function(recon_x, x, mask, mu, log_var)
        else:
            recon_x, mu, log_var = self.model(x)
            loss, rec_loss, kl_loss = self.model.vae_loss(recon_x, x, mu, log_var)
            
        self.log("val_loss", loss, prog_bar=True)

        if batch_idx == 0:
            n = min(8, x.size(0))
            orig = x[:n]
            # マスクありの場合は masked_x も表示、なければ通常画像と同じものを表示
            masked = masked_x[:n] if self.mask_enabled else x[:n]
            recon = recon_x[:n]
            # [元画像, マスク画像, 復元画像]の順で並べる
            image_list = []
            for i in range(n):
                image_list.extend([orig[i], masked[i], recon[i]])
            grid = vutils.make_grid(torch.stack(image_list), nrow=3, normalize=True, padding=2)
            self.logger.experiment.add_image("Reconstruction", grid, self.global_step)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

@hydra.main(config_path="config", config_name="train_vae", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")
    
    # ロガーの作成
    logger = TensorBoardLogger(save_dir=config.log_dir, name="VAE")
    
    # データモジュールの作成
    data = get_data_module(config.data)
    data.setup()
    
    model = VAETrainer(vae_cfg=config.model.vae,
                       learning_rate=config.learning_rate,
                       mask_enabled=config.mask_enabled,
                       mask_ratio=config.mask_ratio,
                       patch_size=config.patch_size)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.save_ckpt_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=1.0
    )
    
    trainer.fit(model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())

if __name__ == "__main__":
    main()
