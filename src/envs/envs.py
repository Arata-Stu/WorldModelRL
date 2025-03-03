from omegaconf import DictConfig
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from envs.car_racing import CarRacingWithInfoWrapper

def get_env(env_cfg: DictConfig):

    width, height = env_cfg.img_size, env_cfg.img_size

    env = gym.make(env_cfg.name, render_mode=env_cfg.render_mode)
    env = TimeLimit(env, max_episode_steps=env_cfg.num_steps)

    if env_cfg.name == "CarRacing-v3":
        return CarRacingWithInfoWrapper(env, width=width, height=height)
    else:
        raise NotImplementedError