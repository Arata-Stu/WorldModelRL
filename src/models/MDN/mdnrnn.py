import torch
import torch.nn as nn
from omegaconf import DictConfig
from .mdn import MDN
# from src.utils.timers import CudaTimer as Timer
from src.utils.timers import TimerDummy as Timer

class MDNRNN(nn.Module):
    """
    MDN-RNNの実装です。
    LSTMで入力系列の文脈情報を抽出し、その出力を用いてMDNで次の潜在変数の分布パラメータを予測します。

    Args:
        mdnrnn_cfg (DictConfig): Hydraを用いた設定（hidden_size, num_layers, num_mixturesなど）
        latent_dim (int): VAEの潜在空間の次元数
        action_dim (int): 行動空間の次元数（必要な場合、入力に含める）
    """
    def __init__(self, mdnrnn_cfg: DictConfig, latent_dim: int, action_dim: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.input_dim = latent_dim + action_dim
        self.hidden_size = mdnrnn_cfg.hidden_size
        self.num_layers = mdnrnn_cfg.num_layers
        self.num_mixtures = mdnrnn_cfg.num_mixtures
        
        # LSTMレイヤー
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        # MDNクラスを利用
        self.mdn = MDN(input_dim=self.hidden_size, latent_dim=self.latent_dim, num_mixtures=self.num_mixtures)
    
    def forward(self, inputs: torch.Tensor, hidden=None):
        """
        Args:
            inputs (torch.Tensor): 入力テンソル、形状は (batch, seq_len, input_dim)
            hidden (optional): LSTMの初期隠れ状態
        
        Returns:
            hidden: LSTMの最終的な隠れ状態
            (pi, mu, sigma): MDNの出力パラメータ
        """
        with Timer(device=inputs.device, timer_name="MDNRNN: forward"):
            if hidden is None:
                lstm_out, hidden = self.lstm(inputs)
            else:
                lstm_out, hidden = self.lstm(inputs, hidden)
            pi, mu, sigma = self.mdn(lstm_out)
        return pi, mu, sigma, hidden
    
    def init_weights(self):
        """LSTMとMDNの重みを適切に初期化"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # 入力から隠れ層の重み
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # 隠れ層から隠れ層の重み
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:  # バイアス
                param.data.fill_(0)
