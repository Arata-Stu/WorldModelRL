import torch
import torch.nn as nn
import torch.nn.init as init
# from src.utils.timers import CudaTimer as Timer
from src.utils.timers import TimerDummy as Timer

class Controller(nn.Module):
    """
    Controllerの実装です。
    MLP（全結合ネットワーク）で、VAEの潜在表現やMDN-RNNの出力などから環境への行動を予測します。

    Args:
        input_dim (int): 入力の次元数（例：VAEの潜在表現 + MDN-RNNの状態など）
        output_dim (int): 出力の次元数（環境の行動次元）
        hidden_dims (list, optional): 隠れ層の次元数リスト。デフォルトは [128, 64]
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [128, 64], ckpt_path: str = None):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        # 最終出力層
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)
        
        # 重みの初期化を適用
        self.apply(self._initialize_weights)

        if ckpt_path is not None:
            self.load_weights(ckpt_path)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力テンソルから行動を予測します。

        Args:
            x (torch.Tensor): 入力テンソル、形状は (batch, input_dim)

        Returns:
            torch.Tensor: 予測された行動、形状は (batch, output_dim)
        """
        with Timer(device=x.device, timer_name="Controller: forward"):
            action = self.model(x)
        return action
    
    def _initialize_weights(self, module):
        """ 重みとバイアスの適切な初期化 """
        if isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)

    def load_weights(self, path: str, strict: bool = True):
        if path.endswith(".pth"):
            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict, strict=strict)
            print(f"Loaded .pth weights from {path}")
        elif path.endswith(".ckpt"):
            checkpoint = torch.load(path, map_location=self.device)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                self.load_state_dict(new_state_dict, strict=strict)
                print(f"Loaded .ckpt weights from {path}")
            else:
                raise ValueError(f"Invalid .ckpt file format: {path}")
        else:
            raise ValueError(f"Unsupported file format: {path}")
