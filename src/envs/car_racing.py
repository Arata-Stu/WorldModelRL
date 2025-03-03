import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CarRacingWithInfoWrapper(gym.Wrapper):
    def __init__(self, env, width=128, height=128):
        super().__init__(env)
        self.width = width
        self.height = height

        # 画像 + 車両情報 (速度, 角度, 位置) を含める
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
            "speed": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),  # (速度X, 速度Y)
            "angular_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "position": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),  # (X座標, Y座標)
            "angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)  # ラジアン角
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_observation(obs), reward, terminated, truncated, info

    def _process_observation(self, obs):
        """ 画像をリサイズし、車両情報を追加 """
        resized_image = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        car = self.env.unwrapped.car

        car_state = {
            "image": resized_image,
            "speed": np.array(car.hull.linearVelocity),  # (vx, vy)
            "angular_velocity": np.array([car.hull.angularVelocity]),
            "position": np.array(car.hull.position),  # (x, y)
            "angle": np.array([car.hull.angle])  # 向き (ラジアン)
        }
        return car_state
