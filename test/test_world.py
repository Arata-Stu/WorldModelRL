import sys
sys.path.append("../")

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import cv2
import numpy as np

from src.envs.envs import get_env
from src.models.world_model import WorldModel

@hydra.main(config_path="../config", config_name="test_world", version_base="1.2")
def main(config: DictConfig):
    # 設定の確認
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 環境の初期化と初期観測画像の取得
    env = get_env(config.envs)
    obs, info = env.reset()  # 初期画像取得
    obs_img = obs["image"]  # RGB形式の画像
    
    # モデルの初期化
    action_dim = 3
    model = WorldModel(cfg=config.model, action_dim=action_dim)
    model.to(device)
    
    # 初期画像を OpenCV 用に変換（RGB -> BGR）
    obs_img_bgr = cv2.cvtColor(obs_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Dream Visualization", obs_img_bgr)
    cv2.waitKey(1)
    
    step = 0
    while step < config.num_steps:
        # HWC numpy (RGB) -> CHW torch tensor に変換し、バッチ次元を追加
        obs_img_tensor = torch.tensor(obs_img, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255

        # Step 1: VAEで観測画像から潜在表現 z を取得
        z = model.vae.obs_to_z(obs_img_tensor)

        # Step 2: ランダムな行動を生成（例：-1〜1の一様分布）
        random_action = torch.rand((1, action_dim), dtype=torch.float32, device=device) * 2 - 1

        # Step 3: MDN-RNN の入力を作成（潜在表現とランダム行動を結合）
        rnn_input = torch.cat([z, random_action], dim=-1).unsqueeze(1)
        pi, mu, sigma, lstm_out, hidden = model.mdn_rnn(rnn_input, model.state)

        # Step 4: MDNから次の潜在状態をサンプリング
        next_z = model.mdn_rnn.mdn.sample(pi, mu, sigma)

        # Step 5: 報酬予測（オプション）
        reward_input = torch.cat([next_z, lstm_out], dim=-1)
        next_reward = model.reward_predictor(reward_input)

        # Step 6: VAEのデコーダーで画像を再構成し、可視化用に整形
        recon_img = model.vae.decode(next_z)
        recon_img_np = recon_img.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        recon_img_uint8 = (recon_img_np * 255).clip(0, 255).astype(np.uint8)
        recon_img_bgr = cv2.cvtColor(recon_img_uint8, cv2.COLOR_RGB2BGR)

        cv2.imshow("Dream Visualization", recon_img_bgr)
        key = cv2.waitKey(100)
        if key == 27:
            break

        # 次のステップの入力画像として再構成画像を利用（必要に応じて前処理を調整）
        obs_img = recon_img_uint8

        step += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
