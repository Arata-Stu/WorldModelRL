import torch
import torch.nn as nn
from omegaconf import DictConfig
from .mdn import MDN
from src.utils.timers import TimerDummy as Timer

class MDNRNN(nn.Module):
    def __init__(self, mdnrnn_cfg: DictConfig, latent_dim: int, action_dim: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.input_dim = latent_dim + action_dim
        self.hidden_size = mdnrnn_cfg.hidden_size
        self.num_layers = mdnrnn_cfg.num_layers
        self.num_mixtures = mdnrnn_cfg.num_mixtures
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        self.mdn = MDN(input_dim=self.hidden_size, latent_dim=self.latent_dim, num_mixtures=self.num_mixtures)
        
        self.init_weights()
        
        if mdnrnn_cfg.ckpt_path is not None:
            self.load_weights(mdnrnn_cfg.ckpt_path)
    
    def forward(self, inputs: torch.Tensor, hidden=None):
        with Timer(device=inputs.device, timer_name="MDNRNN: forward"):
            if hidden is None:
                lstm_out, hidden = self.lstm(inputs)
            else:
                lstm_out, hidden = self.lstm(inputs, hidden)
            pi, mu, sigma = self.mdn(lstm_out)
        return pi, mu, sigma, hidden
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
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