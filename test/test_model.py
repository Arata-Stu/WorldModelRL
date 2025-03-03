import sys
sys.path.append('..')

import torch
from omegaconf import OmegaConf

from src.models.world_model import WorldModel
from src.models.MDN.mdnrnn import MDNRNN

yaml_path = '../config/model/default.yaml'
cfg = OmegaConf.load(yaml_path)

OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
print("------ Configuration ------")
print(OmegaConf.to_yaml(cfg))
print("---------------------------")
model = WorldModel(cfg=cfg, action_dim=3)

sample_img_tensor = torch.randn(1, 3, 64, 64)
z, action  = model(sample_img_tensor)
print("------ Output ------")
print(f"z: {z.shape}")
print(f"action: {action.shape}")

yaml_path = "../config/model/mdn/mdnrnn.yaml"
mdnrnn_cfg = OmegaConf.load(yaml_path)
latent_dim = 32
action_dim = 3
model = MDNRNN(mdnrnn_cfg, latent_dim=latent_dim, action_dim=action_dim)

sample_latent_seq = torch.randn(1, 20, latent_dim)
sample_action_seq = torch.randn(1, 20, action_dim)
## zとactionを結合してmdnrnnに入力
rnn_input = torch.cat([sample_latent_seq, sample_action_seq], dim=-1)
pi, mu, sigma, hidden = model(rnn_input)
print("------ Output ------")