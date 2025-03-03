import os
import numpy as np
import hydra
import h5py
from omegaconf import DictConfig, OmegaConf
from concurrent.futures import ProcessPoolExecutor

from src.envs.envs import get_env

@hydra.main(config_path="config", config_name="collect_data", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")

    os.makedirs(config.output_dir, exist_ok=True)

    def collect_episode(ep):
        """1つのエピソードを収集して保存"""
        env = get_env(config.envs)  # 各プロセスで環境を独立して作成
        obs, info = env.reset()
        episode_obs = []
        episode_actions = []
        episode_rewards = []

        done = False
        step = 0
        while not done and step < config.num_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if isinstance(obs, dict):
                obs = obs["image"]

            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            step += 1

        # HDF5 形式で保存
        file_path = os.path.join(config.output_dir, f"episode_{ep:03d}.h5")
        with h5py.File(file_path, 'w') as f:
            f.create_dataset("observations", data=np.array(episode_obs), compression="gzip")
            f.create_dataset("actions", data=np.array(episode_actions), compression="gzip")
            f.create_dataset("rewards", data=np.array(episode_rewards), compression="gzip")
        print(f"Episode {ep} saved to {file_path}")

        env.close()
    if config.envs.render_mode == 'human':
        num_workers = 1
    else:
        num_workers = min(config.num_workers, os.cpu_count())  # 設定された worker 数を使用

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(collect_episode, range(config.num_episodes))

if __name__ == "__main__":
    main()
