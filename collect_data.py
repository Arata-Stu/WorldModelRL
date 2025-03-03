import os
import argparse
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

def collect_and_save_data(args):
    # 環境の初期化（render_mode は "rgb_array" を指定）
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=args.max_steps)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for ep in range(args.num_episodes):
        obs, info = env.reset()
        episode_obs = []      # 各ステップの画像データ（例：RGB画像）
        episode_actions = []  # 各ステップの行動
        episode_rewards = []  # 各ステップの報酬
        
        done = False
        step = 0
        while not done and step < args.max_steps:
            # 行動はここではランダムに選択（または手動操作に切り替え可能）
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
        
        # エピソードごとに npz 形式で保存
        episode_data = {
            "observations": np.array(episode_obs),  # 形状: (steps, H, W, C)
            "actions": np.array(episode_actions),     # 形状: (steps, action_dim)
            "rewards": np.array(episode_rewards)        # 形状: (steps,)
        }
        file_path = os.path.join(args.output_dir, f"episode_{ep:03d}.npz")
        np.savez(file_path, **episode_data)
        print(f"Episode {ep} saved to {file_path}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=10, help="収集するエピソード数")
    parser.add_argument("--max_steps", type=int, default=1000, help="エピソードあたりの最大ステップ数")
    parser.add_argument("--output_dir", type=str, required=True, help="データ保存先ディレクトリ")
    args = parser.parse_args()
    collect_and_save_data(args)
