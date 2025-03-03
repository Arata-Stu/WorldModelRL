import os
import numpy as np
import hydra
import h5py

from src.envs.envs import get_env

@hydra.main(config_path="config", config_name="collect_data", version_base="1.2")
def main(args):
    # 環境の初期化
    env = get_env(args.env)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for ep in range(args.num_episodes):
        obs, info = env.reset()
        episode_obs = []      # 各ステップの画像データ（例：RGB画像）
        episode_actions = []  # 各ステップの行動
        episode_rewards = []  # 各ステップの報酬
        
        done = False
        step = 0
        while not done and step < args.num_steps:
            # ランダムに行動を選択
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # obsが dict 型の場合は "image" キーから画像を取得
            if isinstance(obs, dict):
                obs = obs["image"]
            
            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            step += 1
        
        # エピソードごとに HDF5 形式で保存（gzip圧縮を使用、不要なら compression を削除）
        file_path = os.path.join(args.output_dir, f"episode_{ep:03d}.h5")
        with h5py.File(file_path, 'w') as f:
            f.create_dataset("observations", data=np.array(episode_obs), compression="gzip")
            f.create_dataset("actions", data=np.array(episode_actions), compression="gzip")
            f.create_dataset("rewards", data=np.array(episode_rewards), compression="gzip")
        print(f"Episode {ep} saved to {file_path}")
    
    env.close()

if __name__ == "__main__":
    main()
