import torch
import torch.nn as nn
from omegaconf import DictConfig
from src.models.VAE.VAE import get_vae
from src.models.MDN.mdnrnn import MDNRNN
from src.models.Controller.controller import Controller
# from src.utils.timers import CudaTimer as Timer
from src.utils.timers import TimerDummy as Timer

class WorldModel(nn.Module):
    """
    World Model の統合クラス。
    VAE, MDN-RNN, Controller を組み合わせてエージェントの学習を管理。

    Args:
        cfg (DictConfig): Hydraを用いた設定オブジェクト
    """
    def __init__(self, cfg: DictConfig, action_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 各モデルの初期化
        self.vae = get_vae(vae_cfg=cfg.vae).to(self.device)
        self.mdn_rnn = MDNRNN(mdnrnn_cfg=cfg.mdn, latent_dim=cfg.vae.latent_dim, action_dim=action_dim).to(self.device)
        self.controller = Controller(input_dim=cfg.vae.latent_dim + cfg.mdn.hidden_size, 
                                     output_dim=action_dim, 
                                     hidden_dims=cfg.controller.hidden_dims).to(self.device)
        
        # 初期の隠れ状態をクラス属性として保持
        self.state = (torch.zeros((1, 1, self.mdn_rnn.hidden_size)).to(self.device),
                      torch.zeros((1, 1, self.mdn_rnn.hidden_size)).to(self.device))

    def forward(self, obs: torch.Tensor, hidden=None):
        """
        現在の状態(obs)と隠れ状態(hidden)から行動を推論し、
        さらに状態(obs)と行動と隠れ状態から次の状態を予測する。

        Args:
            obs (torch.Tensor): 環境の観測データ（画像など）
            hidden (optional): MDN-RNN の隠れ状態 (LSTM state)

        Returns:
            next_z (torch.Tensor): 予測された次の潜在表現
            next_action (torch.Tensor): 予測された次の行動
        """
        with Timer(device=obs.device, timer_name="WorldModel: forward"):
            # hidden が指定されない場合は、クラスで保持している状態を使用
            if hidden is None:
                hidden = self.state

            # Step 1: VAE で観測を潜在表現 z に変換
            z = self.vae.obs_to_z(obs)
            
            # Step 2: Controller が行動を決定（現在の z と隠れ状態 h_t を利用）
            controller_input = torch.cat([z, hidden[0][0]], dim=1)
            next_action = self.controller(controller_input)
            
            # Step 3: MDN-RNN で次の潜在表現を予測
            rnn_input = torch.cat([z, next_action], dim=-1).unsqueeze(1)  # 潜在変数と行動を結合
            pi, mu, sigma, hidden = self.mdn_rnn(rnn_input, hidden)
            
            # Step 4: 次の潜在表現をサンプリング
            next_z = self.mdn_rnn.mdn.sample(pi, mu, sigma)
            
            # 更新された隠れ状態をクラス属性に保持
            self.state = hidden

        return next_z, next_action




    def save_checkpoint(self, path: str):
        """
        モデルのチェックポイントを保存。

        Args:
            path (str): 保存先のファイルパス
        """
        checkpoint = {
            "vae": self.vae.state_dict(),
            "mdnrnn": self.mdn_rnn.state_dict(),
            "controller": self.controller.state_dict()
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """
        モデルのチェックポイントをロード。

        Args:
            path (str): 読み込むファイルのパス
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(checkpoint["vae"])
        self.mdn_rnn.load_state_dict(checkpoint["mdnrnn"])
        self.controller.load_state_dict(checkpoint["controller"])
        print(f"Checkpoint loaded from {path}")
