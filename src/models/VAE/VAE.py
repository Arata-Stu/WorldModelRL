from omegaconf import DictConfig

from .CNN_VAE import CNN_VAE
from .MaxVIT_VAE import MaxVIT_VAE

def get_vae(vae_cfg: DictConfig):
    latent_dim = vae_cfg.latent_dim
    input_shape = vae_cfg.input_shape

    if vae_cfg.name == 'cnn_vae':
        return CNN_VAE(cnn_cfg=vae_cfg, latent_dim=latent_dim, input_shape=input_shape)
    elif vae_cfg.name == 'maxvit_vae':
        return MaxVIT_VAE(maxvit_cfg=vae_cfg, latent_dim=latent_dim, input_shape=input_shape)
    else:
        raise NotImplementedError(f"name {vae_cfg.name} not implemented")