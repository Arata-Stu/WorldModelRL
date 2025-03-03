import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MDN(nn.Module):
    """
    Mixture Density Network の実装です。
    LSTMなどからの隠れ状態を入力として、混合正規分布のパラメータ（混合係数pi、平均mu、標準偏差sigma）を算出します。

    Args:
        input_dim (int): MDNの入力次元数（例：LSTMの隠れ状態の次元数）。
        latent_dim (int): 潜在空間の次元数。
        num_mixtures (int): 混合成分の数。
    """
    def __init__(self, input_dim: int, latent_dim: int, num_mixtures: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures
        self.mdn_output_dim = num_mixtures * (1 + 2 * latent_dim)
        self.fc = nn.Linear(input_dim, self.mdn_output_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MDNの出力を計算します。

        Args:
            x (torch.Tensor): 入力テンソル、形状は (batch, seq_len, input_dim)

        Returns:
            pi (torch.Tensor): 混合係数、形状 (batch, seq_len, num_mixtures)
            mu (torch.Tensor): 各混合成分の平均、形状 (batch, seq_len, num_mixtures, latent_dim)
            sigma (torch.Tensor): 各混合成分の標準偏差、形状 (batch, seq_len, num_mixtures, latent_dim)
        """
        mdn_out = self.fc(x)  # shape: (batch, seq_len, mdn_output_dim)
        batch, seq_len, _ = x.shape
        # リシェイプして (batch, seq_len, num_mixtures, 1 + 2*latent_dim)
        mdn_out = mdn_out.view(batch, seq_len, self.num_mixtures, 1 + 2 * self.latent_dim)
        
        # 最初の値がpi、その後latent_dim個がmu、残りがsigmaとなる
        pi = mdn_out[..., 0]
        mu = mdn_out[..., 1:1+self.latent_dim]
        sigma = mdn_out[..., 1+self.latent_dim:]
        
        # 混合係数piにsoftmax、sigmaにexpを適用して正の値に変換
        pi = F.softmax(pi, dim=-1)
        sigma = torch.exp(sigma)
        
        return pi, mu, sigma

    def sample(self, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        MDNの出力からサンプリングを行います。

        Args:
            pi (torch.Tensor): 混合係数、形状 (batch, seq_len, num_mixtures)
            mu (torch.Tensor): 各混合成分の平均、形状 (batch, seq_len, num_mixtures, latent_dim)
            sigma (torch.Tensor): 各混合成分の標準偏差、形状 (batch, seq_len, num_mixtures, latent_dim)

        Returns:
            sample (torch.Tensor): サンプリングされた潜在変数、形状 (batch, seq_len, latent_dim)
        """
        batch_size, seq_len, num_mixtures = pi.shape
        latent_dim = mu.shape[-1]  # 取得する潜在次元

        # 混合係数に基づいてサンプリングする混合成分のインデックスを選択
        pi_flat = pi.view(-1, num_mixtures)  # (batch * seq_len, num_mixtures)
        mixture_indices = torch.multinomial(pi_flat, num_samples=1)
        mixture_indices = mixture_indices.view(batch_size, seq_len, 1)  # (batch, seq_len, 1)
        
        # インデックスを mu の次元に合わせる (batch, seq_len, 1, latent_dim)
        indices_expanded = mixture_indices.unsqueeze(-1).expand(-1, -1, -1, latent_dim)
        
        # 選択された混合成分の mu と sigma を取得
        mu_selected = torch.gather(mu, 2, indices_expanded).squeeze(2)  # (batch, seq_len, latent_dim)
        sigma_selected = torch.gather(sigma, 2, indices_expanded).squeeze(2)  # (batch, seq_len, latent_dim)
    
        # 正規分布に従ってサンプリング
        eps = torch.randn_like(mu_selected)
        sample = mu_selected + sigma_selected * eps
        return sample

