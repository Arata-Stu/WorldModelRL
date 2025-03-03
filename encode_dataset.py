import os
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from src.models.VAE.VAE import get_vae  # VAE の実装があると仮定

def encode_and_update_npz(vae, data_dir, output_dir, device="cpu"):
    """
    VAE を用いて npz ファイルの観測データを潜在変数に変換し、元の npz に z を追加して保存。

    Args:
        vae (torch.nn.Module): 事前学習済みの VAE モデル
        data_dir (str): npz ファイルが格納されたディレクトリ
        output_dir (str): 潜在変数を保存するディレクトリ（上書きではなく新規保存）
        device (str): "cpu" または "cuda"
    """
    os.makedirs(output_dir, exist_ok=True)

    # data_dir 内の全ての npz ファイルを処理
    for file_name in sorted(os.listdir(data_dir)):  # ソートして順番を統一
        if file_name.endswith(".npz"):
            data_path = os.path.join(data_dir, file_name)
            output_path = os.path.join(output_dir, file_name)  # 元と同じファイル名で保存

            # npz ファイルをロード
            data = np.load(data_path)
            observations = data["observations"]  # 形状: (steps, H, W, C)
            actions = data["actions"]  # 形状: (steps, action_dim)
            rewards = data["rewards"]  # 形状: (steps,)
            data.close()

            # Tensor に変換（(N, H, W, C) → (N, C, H, W)）
            obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)

            # 必要に応じて正規化（例：0-255 → 0-1）
            obs_tensor = obs_tensor / 255.0

            # VAE を用いて潜在表現 z を取得
            with torch.no_grad():
                z = vae.obs_to_z(obs_tensor)  # 形状: (steps, latent_dim)

            latent_data = z.cpu().numpy()  # NumPy に変換

            # 元の npz に `latent` を追加して新しく保存
            np.savez(output_path, observations=observations, actions=actions, rewards=rewards, latent=latent_data)
            print(f"Updated {file_name} with latent data -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="npz ファイルが格納されたディレクトリ")
    parser.add_argument("--output_dir", type=str, required=True, help="潜在変数を保存するディレクトリ")
    parser.add_argument("--vae_yaml_path", type=str, required=True, help="VAE の設定 yaml ファイルのパス")
    args = parser.parse_args()

    # VAE の設定ファイルを読み込み、モデルを構築・ロード
    vae_cfg = OmegaConf.load(args.vae_yaml_path)
    vae = get_vae(vae_cfg=vae_cfg)
    vae.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    encode_and_update_npz(vae, args.data_dir, args.output_dir, device=device)
