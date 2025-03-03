from typing import Tuple
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .base import BaseVAE
from .timm.maxvit_encoder import MaxxVitEncoder
from .timm.maxvit_decoder import MaxxVitDecoder
from src.utils.timers import CudaTimer as Timer
# from src.utils.timers import TimerDummy as Timer

class MaxVIT_VAE(BaseVAE):
    """
    MaxVIT Variational Autoencoder (VAE) のクラス実装。
    
    画像を潜在ベクトルに圧縮し、VAE の特性として潜在空間に確率的なサンプリングを導入。
    """
    def __init__(self, maxvit_cfg: DictConfig, latent_dim: int = 512, input_shape: Tuple[int, int, int] = (3, 64, 64)):
        # BaseVAE の初期化（latent_dim のみ使用）
        super(MaxVIT_VAE, self).__init__(latent_dim)

        # エンコーダの初期化
        in_chans, height, width = input_shape
        self.encoder = MaxxVitEncoder(maxvit_cfg=maxvit_cfg,
                                      img_size=width,
                                      in_chans=in_chans,
                                      drop_path_rate=maxvit_cfg.drop_path_rate)

        # エンコーダの出力チャネル数・特徴マップサイズを取得
        encoder_out_chs = self.encoder.feature_info[-1]['num_chs']
        encoder_out_size = width // self.encoder.feature_info[-1]['reduction']
        self.latent_shape = (encoder_out_chs, encoder_out_size, encoder_out_size)

        # Flatten: 特徴マップ → 1D 潜在ベクトル
        self.flatten = nn.Flatten()
        latent_dim_from_encoder = encoder_out_chs * encoder_out_size * encoder_out_size

        # 潜在変数の次元は、指定された latent_dim とエンコーダから得られる次元数の小さい方に合わせる
        self.latent_size = min(latent_dim, latent_dim_from_encoder)

        # 平均と分散を学習する全結合層
        self.fc_mu = nn.Linear(latent_dim_from_encoder, self.latent_size)
        self.fc_log_var = nn.Linear(latent_dim_from_encoder, self.latent_size)

        # デコーダの入力層
        self.fc_decode = nn.Linear(self.latent_size, latent_dim_from_encoder)

        # デコーダの初期化
        self.decoder = MaxxVitDecoder(maxvit_cfg, in_chans=encoder_out_chs, input_size=encoder_out_size)

        if maxvit_cfg.ckpt_path:
            self.load_weights(maxvit_cfg.ckpt_path)
        
        
    def encode(self, x):
        with Timer(device=x.device, timer_name="VAE: encode"):
            z = self.encoder(x)            # 出力形状: (B, C, H, W)
            z = self.flatten(z)            # (B, C*H*W)
            mu = self.fc_mu(z)             # (B, latent_dim)
            log_var = self.fc_log_var(z)   # (B, latent_dim)
        return mu, log_var

    def decode(self, z):
        with Timer(device=z.device, timer_name="VAE: decode"):
            z = self.fc_decode(z)                  # (B, C*H*W)
            z = z.view(-1, *self.latent_shape)       # (B, C, H, W)
            x_recon = self.decoder(z)
            x_recon = torch.sigmoid(x_recon)  # 出力を [0,1] の範囲に正規化
        return x_recon

    def forward(self, x):
        with Timer(device=x.device, timer_name="VAE: encode + decode"):
            mu, log_var = self.encode(x)
            # BaseVAE の latent() を利用して再パラメータ化を実施
            z = self.latent(mu, log_var)
            x_recon = self.decode(z)
        return x_recon, mu, log_var
