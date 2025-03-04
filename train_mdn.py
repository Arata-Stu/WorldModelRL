import os
import hydra
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import  get_data_module
from src.models.MDN.mdnrnn import MDNRNN
from src.models.reward.reward import RewardPredictor

class WorldModelModule(pl.LightningModule):
    """
    MDN-RNN と RewardPredictor を統合した学習モジュール。
    次状態予測と報酬予測を同時に行う。
    """
    def __init__(self, mdnrnn_cfg: DictConfig, latent_dim: int, action_dim: int = 0, 
                 reward_cfg: DictConfig = None, learning_rate: float = 1e-3, reward_loss_weight: float = 1.0):
        """
        Args:
            mdnrnn_cfg (DictConfig): MDN-RNN の設定（hidden_size, num_layers, num_mixtures など）
            latent_dim (int): VAE の潜在空間の次元数
            action_dim (int): 行動空間の次元数
            reward_cfg (DictConfig): RewardPredictor の設定（hidden_dims など）
            learning_rate (float): 学習率
            reward_loss_weight (float): 報酬予測の損失にかける重み
        """
        super().__init__()
        # self.save_hyperparameters(ignore=['reward_cfg'])
        self.learning_rate = learning_rate
        self.reward_loss_weight = reward_loss_weight

        # MDN-RNN の初期化
        self.mdnrnn = MDNRNN(mdnrnn_cfg, latent_dim, action_dim)

        # RewardPredictor の初期化
        # reward_cfg.hidden_dims はリストで指定 (例: [128, 64])
        self.reward_predictor = RewardPredictor(input_dim=latent_dim, hidden_dims=reward_cfg.hidden_dims)

    def forward(self, latent_seq, action_seq, hidden=None):
        """
        forward 内で MDN-RNN による次状態予測と、次状態からの報酬予測を行う。

        Returns:
            pi, mu, sigma: MDN の出力パラメータ（次状態予測用）
            next_z: MDN-RNN からサンプリングされた次状態
            predicted_reward: RewardPredictor により推論された報酬
            hidden: LSTM の隠れ状態
        """
        # MDN-RNN の入力: 潜在状態と行動を結合
        inputs = torch.cat([latent_seq, action_seq], dim=-1)
        pi, mu, sigma, hidden = self.mdnrnn(inputs, hidden)
        next_z = self.mdnrnn.mdn.sample(pi, mu, sigma)
        # 次状態 next_z から報酬予測
        predicted_reward = self.reward_predictor(next_z)
        return pi, mu, sigma, next_z, predicted_reward, hidden

    def training_step(self, batch, batch_idx):
        latent_seq, action_seq, target_z, target_reward = batch
        pi, mu, sigma, next_z, pred_reward, _ = self(latent_seq, action_seq)

        # MDN-RNN の損失（次状態予測誤差）
        loss_mdn = self.mdnrnn.loss(target_z, pi, mu, sigma)
        # 報酬予測の損失（MSE）
        loss_reward = F.mse_loss(pred_reward, target_reward)
        loss = loss_mdn + self.reward_loss_weight * loss_reward

        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_mdn", loss_mdn, prog_bar=True)
        self.log("loss_reward", loss_reward, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        latent_seq, action_seq, target_z, target_reward = batch
        pi, mu, sigma, next_z, pred_reward, _ = self(latent_seq, action_seq)
        loss_mdn = self.mdnrnn.loss(target_z, pi, mu, sigma)
        loss_reward = F.mse_loss(pred_reward, target_reward)
        loss = loss_mdn + self.reward_loss_weight * loss_reward

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss_mdn", loss_mdn, prog_bar=True)
        self.log("val_loss_reward", loss_reward, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)




@hydra.main(config_path="config", config_name="train_mdn", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")

    # ロガーの作成
    logger = TensorBoardLogger(save_dir=config.log_dir, name="WorldModel")

    # データモジュールの作成
    data_module =get_data_module(config.data)
    data_module.setup()

    # データセットから latent_dim, action_dim を取得
    sample = data_module.train_dataset[0]
    latent_dim = sample[0].shape[-1]
    action_dim = sample[1].shape[-1]

    model = WorldModelModule(
        mdnrnn_cfg=config.model.mdn,
        latent_dim=latent_dim,
        action_dim=action_dim,
        reward_cfg=config.model.reward,  # RewardPredictor 用の設定
        learning_rate=config.learning_rate,
        reward_loss_weight=config.reward_loss_weight
    )

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
    
    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())

if __name__ == "__main__":
    main()