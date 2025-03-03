import os
import argparse
import numpy as np
import torch
import h5py
from omegaconf import OmegaConf
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

    # data_dir 内の全ての h5 ファイルを処理（拡張子が .h5 または .hdf5）
    for file_name in sorted(os.listdir(data_dir)):
        if file_name.endswith(".h5") or file_name.endswith(".hdf5"):
            data_path = os.path.join(data_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # h5 ファイルを読み込み
            with h5py.File(data_path, 'r') as f:
                observations = f["observations"][()]  # 形状: (steps, H, W, C)
                actions = f["actions"][()]            # 形状: (steps, action_dim)
                rewards = f["rewards"][()]            # 形状: (steps,)

            # Tensor へ変換し、(N, H, W, C) → (N, C, H, W) に変換
            obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
            # 必要に応じて正規化（例：0-255 → 0-1）
            obs_tensor = obs_tensor / 255.0

            # VAE を用いて潜在表現 z を取得
            with torch.no_grad():
                z = vae.obs_to_z(obs_tensor)  # 形状: (steps, latent_dim)
            latent_data = z.cpu().numpy()

            # latent を追加して新しい h5 ファイルとして保存（gzip圧縮を使用）
            with h5py.File(output_path, 'w') as f:
                f.create_dataset("observations", data=observations, compression="gzip")
                f.create_dataset("actions", data=actions, compression="gzip")
                f.create_dataset("rewards", data=rewards, compression="gzip")
                f.create_dataset("latent", data=latent_data, compression="gzip")
            print(f"Updated {file_name} with latent data -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="h5 ファイルが格納されたディレクトリ")
    parser.add_argument("--output_dir", type=str, required=True, help="潜在変数を保存するディレクトリ")
    parser.add_argument("--vae_yaml_path", type=str, required=True, help="VAE の設定 yaml ファイルのパス")
    args = parser.parse_args()

    # VAE の設定ファイルを読み込み、モデルを構築・ロード
    vae_cfg = OmegaConf.load(args.vae_yaml_path)
    vae = get_vae(vae_cfg=vae_cfg)
    vae.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    encode_and_update_h5(vae, args.data_dir, args.output_dir, device=device)
