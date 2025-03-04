import sys
sys.path.append("../")

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import cv2
import numpy as np

from src.envs.envs import get_env
from src.models.world_model import WorldModel

# --- ヘルパー関数: モジュールのパラメータを平坦化・更新 ---
def get_flat_params(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([param.data.view(-1) for param in module.parameters()])

def set_flat_params(module: torch.nn.Module, flat_params: torch.Tensor):
    pointer = 0
    for param in module.parameters():
        num_param = param.numel()
        param.data.copy_(flat_params[pointer:pointer+num_param].view_as(param))
        pointer += num_param

# --- シミュレーション関数 ---
def simulate_dream(model: WorldModel, initial_obs: np.ndarray, simulation_steps: int, device: torch.device) -> float:
    # MDN-RNNの隠れ状態をゼロで初期化
    hidden_size = model.mdn_rnn.hidden_size
    model.state = (torch.zeros((1, 1, hidden_size), device=device),
                   torch.zeros((1, 1, hidden_size), device=device))
    
    # シミュレーション開始：初期観測画像からスタート
    obs_img = initial_obs.copy()
    cumulative_reward = 0.0

    for step in range(simulation_steps):
        # 観測画像 -> テンソル変換
        obs_img_tensor = torch.tensor(obs_img, dtype=torch.float32, device=device) \
                             .permute(2, 0, 1).unsqueeze(0) / 255.0
        # VAEで潜在表現 z を取得
        z = model.vae.obs_to_z(obs_img_tensor)
        
        # Controllerで行動決定（MDN-RNNの隠れ状態の一部と結合）
        controller_input = torch.cat([z, model.state[0][0]], dim=1)
        action = model.controller(controller_input)
        
        # MDN-RNNで次の潜在状態を予測
        rnn_input = torch.cat([z, action], dim=-1).unsqueeze(1)
        pi, mu, sigma, lstm_out, hidden = model.mdn_rnn(rnn_input, model.state)
        model.state = hidden  # 隠れ状態を更新
        
        # MDNから次の潜在状態をサンプリング
        next_z = model.mdn_rnn.mdn.sample(pi, mu, sigma)
        
        # RewardPredictorで報酬を予測
        reward_input = torch.cat([next_z, lstm_out], dim=-1)
        reward = model.reward_predictor(reward_input)
        cumulative_reward += reward.item()
        
        # VAEのデコーダーで画像を再構成→次の観測とする
        recon_img = model.vae.decode(next_z)
        obs_img = (recon_img.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255) \
                     .clip(0, 255).astype(np.uint8)
    
    return cumulative_reward

@hydra.main(config_path="config", config_name="test_controller", version_base="1.2")
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
    initial_obs = obs["image"]  # RGB形式の画像

    # World Modelの初期化
    action_dim = 3
    model = WorldModel(cfg=config.model, action_dim=action_dim)
    model.to(device)
    
    # 初期画像の表示（参考用）
    obs_img_bgr = cv2.cvtColor(initial_obs, cv2.COLOR_RGB2BGR)
    cv2.imshow("Initial Observation", obs_img_bgr)
    cv2.waitKey(1)
    
    # 進化戦略（ES）のハイパーパラメータ
    num_iterations = config.es.num_iterations if 'es' in config and 'num_iterations' in config.es else 100
    population_size = config.es.population_size if 'es' in config and 'population_size' in config.es else 20
    sigma = config.es.sigma if 'es' in config and 'sigma' in config.es else 0.1
    learning_rate = config.es.learning_rate if 'es' in config and 'learning_rate' in config.es else 0.01
    simulation_steps = config.es.simulation_steps if 'es' in config and 'simulation_steps' in config.es else config.num_steps

    # 現在のcontrollerパラメータを平坦化して基準とする
    baseline_params = get_flat_params(model.controller)
    
    for iteration in range(num_iterations):
        rewards = []
        noise_list = []
        
        # 個体群の評価
        for i in range(population_size):
            # ノイズをサンプリング
            noise = torch.randn_like(baseline_params) * sigma
            noise_list.append(noise)
            
            # 候補パラメータ = 基準パラメータ + ノイズ
            candidate_params = baseline_params + noise
            set_flat_params(model.controller, candidate_params)
            
            # dreamシミュレーションを実行し、累積報酬を取得
            reward = simulate_dream(model, initial_obs, simulation_steps, device)
            rewards.append(reward)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        # 報酬の正規化（標準化）で評価のばらつきを抑制
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # 勾配推定の計算
        grad_estimate = torch.zeros_like(baseline_params)
        for i in range(population_size):
            grad_estimate += rewards_normalized[i] * noise_list[i]
        grad_estimate /= population_size
        
        # 基準パラメータの更新
        baseline_params = baseline_params + learning_rate * grad_estimate
        set_flat_params(model.controller, baseline_params)
        
        avg_reward = rewards_tensor.mean().item()
        print(f"Iteration {iteration+1}/{num_iterations}, Average Reward: {avg_reward}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
