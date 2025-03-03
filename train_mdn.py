import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.MDN.mdnrnn import MDNRNN
from src.data.dataset import get_data_module 


class MDNRNNModule(pl.LightningModule):
    """
    PyTorch Lightning を用いた MDN-RNN のトレーニングモジュール
    """

    def __init__(self, mdnrnn_cfg: DictConfig, latent_dim: int, action_dim: int = 0, learning_rate: float = 1e-3):
        """
        Args:
            mdnrnn_cfg (DictConfig): Hydra 設定（hidden_size, num_layers, num_mixtures など）
            latent_dim (int): VAE の潜在空間の次元数
            action_dim (int): 行動空間の次元数（必要な場合、入力に含める）
            learning_rate (float): 学習率
        """
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # MDN-RNN モデルの初期化
        self.model = MDNRNN(mdnrnn_cfg, latent_dim, action_dim)

    def forward(self, latent_seq, action_seq, hidden=None):
        """
        MDNRNN の forward 関数

        Args:
            latent_seq (torch.Tensor): (batch, seq_length, latent_dim)
            action_seq (torch.Tensor): (batch, seq_length, action_dim)
            hidden (optional): LSTM の隠れ状態

        Returns:
            pi, mu, sigma: MDN の出力パラメータ
            hidden: LSTM の隠れ状態
        """
        inputs = torch.cat([latent_seq, action_seq], dim=-1)  # (batch, seq_length, latent_dim + action_dim)
        pi, mu, sigma, hidden = self.model(inputs, hidden)
        return pi, mu, sigma, hidden

    def training_step(self, batch, batch_idx):
        latent_seq, action_seq, target_z = batch
        pi, mu, sigma, _ = self(latent_seq, action_seq)

        loss = self.mdn_loss(pi, mu, sigma, target_z)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        latent_seq, action_seq, target_z = batch
        pi, mu, sigma, _ = self(latent_seq, action_seq)
        
        loss = self.mdn_loss(pi, mu, sigma, target_z)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def mdn_loss(self, pi, mu, sigma, target_z):
        """
        MDN の負の対数尤度 (Negative Log Likelihood) 損失関数
        """
        target_z = target_z.unsqueeze(2)  # (batch, seq_length, 1, latent_dim)
        prob = torch.exp(-0.5 * ((target_z - mu) / sigma) ** 2) / (sigma * (2 * np.pi) ** 0.5)
        prob_weighted = pi * torch.prod(prob, dim=-1)
        loss = -torch.log(torch.sum(prob_weighted, dim=-1) + 1e-8)
        return loss.mean()


@hydra.main(config_path="config", config_name="train_mdn", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ロガーの作成
    logger = TensorBoardLogger(save_dir=config.log_dir, name="MDN")

    # データモジュールの作成
    data = get_data_module(config.data)
    data.setup()

    latent_dim = data.train_dataset[0][0].shape[-1]
    action_dim = data.train_dataset[0][1].shape[-1]

    model = MDNRNNModule(config.model.mdn,
                         latent_dim=latent_dim,
                         action_dim=action_dim,
                         learning_rate=config.learning_rate)
    
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