import torch
import torch.nn as nn
import torch.nn.init as init
from src.utils.timers import CudaTimer as Timer
# from src.utils.timers import TimerDummy as Timer  

class RewardPredictor(nn.Module):
    """
    簡単なMLPによる報酬予測ネットワーク。
    入力は潜在状態 (latent) で、出力はスカラーの報酬予測値となる。
    """
    def __init__(self, input_dim: int, hidden_dims: list, ckpt_path: str = None):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))  # 出力はスカラー
        self.model = nn.Sequential(*layers)

        self.apply(self._initialize_weights)

        if ckpt_path is not None:
            self.load_weights(ckpt_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with Timer(device=x.device, timer_name="RewardPredictor: forward"):
            return self.model(x)
        
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
        