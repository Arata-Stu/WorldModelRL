import os
import torch
import h5py
import hydra
from omegaconf import DictConfig, OmegaConf
from src.models.VAE.VAE import get_vae  # VAE の実装があると仮定

def encode_and_update_h5(vae, data_dir, output_dir, device="cpu"):
    """
    VAE を用いて h5 ファイルの観測データを潜在変数に変換し、
    元の h5 ファイルに latent を追加して保存する関数。

    Args:
        vae (torch.nn.Module): 事前学習済みの VAE モデル
        data_dir (str): h5 ファイルが格納されたディレクトリ
        output_dir (str): 潜在変数を保存するディレクトリ（上書きではなく新規保存）
        device (str): "cpu" または "cuda"
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_name in sorted(os.listdir(data_dir)):
        if file_name.endswith(".h5") or file_name.endswith(".hdf5"):
            data_path = os.path.join(data_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            with h5py.File(data_path, 'r') as f:
                observations = f["observations"][()]
                actions = f["actions"][()]
                rewards = f["rewards"][()]

            obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
            obs_tensor = obs_tensor.permute(0, 3, 1, 2) / 255.0

            with torch.no_grad():
                z = vae.obs_to_z(obs_tensor)
            latent_data = z.cpu().numpy()

            with h5py.File(output_path, 'w') as f:
                f.create_dataset("observations", data=observations, compression="gzip")
                f.create_dataset("actions", data=actions, compression="gzip")
                f.create_dataset("rewards", data=rewards, compression="gzip")
                f.create_dataset("latent", data=latent_data, compression="gzip")
            print(f"Updated {file_name} with latent data -> {output_path}")

@hydra.main(config_path="config", config_name="encode", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    vae_cfg = OmegaConf.load(cfg.model.vae)
    vae = get_vae(vae_cfg=vae_cfg)
    vae.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    
    encode_and_update_h5(vae, cfg.data_dir, cfg.output_dir, device=device)

if __name__ == "__main__":
    main()
