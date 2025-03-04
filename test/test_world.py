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
        obs_img_tensor = torch.tensor(obs_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # WorldModel の forward を呼び出し、潜在状態や行動、報酬を予測
        pred_z, pred_action, pred_reward = model(obs_img_tensor)
        
        # VAE のデコーダーを使って潜在状態から画像を再構成
        recon_img = model.vae.decode(pred_z)
        
        # torch tensor (CHW) -> numpy (HWC) への変換
        recon_img_np = recon_img.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        
        # データスケールが [0, 1] であれば [0, 255] に変換、また OpenCV 用に RGB -> BGR 変換
        recon_img_uint8 = (recon_img_np * 255).clip(0, 255).astype(np.uint8)
        recon_img_bgr = cv2.cvtColor(recon_img_uint8, cv2.COLOR_RGB2BGR)
        
        # 画像をウィンドウに表示（100ms 待機）
        cv2.imshow("Dream Visualization", recon_img_bgr)
        key = cv2.waitKey(100)
        # ESCキーで中断可能
        if key == 27:
            break
        
        # 次のステップの入力画像として再構成画像を利用（必要に応じて前処理を調整）
        obs_img = recon_img_uint8
        
        step += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
