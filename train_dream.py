import os
import sys
sys.path.append("../")

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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
    hidden_size = model.mdn_rnn.hidden_size
    model.state = (torch.zeros((1, 1, hidden_size), device=device),
                   torch.zeros((1, 1, hidden_size), device=device))
    
    obs_img = initial_obs.copy()
    cumulative_reward = 0.0

    for step in range(simulation_steps):
        obs_img_tensor = torch.tensor(obs_img, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
        z = model.vae.obs_to_z(obs_img_tensor)
        controller_input = torch.cat([z, model.state[0][0]], dim=1)
        action = model.controller(controller_input)
        rnn_input = torch.cat([z, action], dim=-1).unsqueeze(1)
        pi, mu, sigma, lstm_out, hidden = model.mdn_rnn(rnn_input, model.state)
        model.state = hidden
        
        next_z = model.mdn_rnn.mdn.sample(pi, mu, sigma)
        reward_input = torch.cat([next_z, lstm_out], dim=-1)
        reward = model.reward_predictor(reward_input)
        cumulative_reward += reward.item()
        
        recon_img = model.vae.decode(next_z)
        obs_img = (recon_img.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    
    return cumulative_reward

@hydra.main(config_path="config", config_name="train_dream", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = get_env(config.envs)
    obs, info = env.reset()
    initial_obs = obs["image"]

    action_dim = 3
    model = WorldModel(cfg=config.model, action_dim=action_dim)
    model.to(device)
    
    obs_img_bgr = cv2.cvtColor(initial_obs, cv2.COLOR_RGB2BGR)
    cv2.imshow("Initial Observation", obs_img_bgr)
    cv2.waitKey(1)
    
    num_iterations = config.es.num_iterations if 'es' in config and 'num_iterations' in config.es else 100
    population_size = config.es.population_size if 'es' in config and 'population_size' in config.es else 20
    sigma = config.es.sigma if 'es' in config and 'sigma' in config.es else 0.1
    learning_rate = config.es.learning_rate if 'es' in config and 'learning_rate' in config.es else 0.01
    simulation_steps = config.es.simulation_steps if 'es' in config and 'simulation_steps' in config.es else config.num_steps

    save_dir = config.save_dir if 'es' in config and 'save_dir' in config.es else "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))

    baseline_params = get_flat_params(model.controller)
    best_params = baseline_params.clone()
    best_reward = float('-inf')
    
    for iteration in range(num_iterations):
        rewards = []
        noise_list = []
        
        for i in range(population_size):
            noise = torch.randn_like(baseline_params) * sigma
            noise_list.append(noise)
            
            candidate_params = baseline_params + noise
            set_flat_params(model.controller, candidate_params)
            
            reward = simulate_dream(model, initial_obs, simulation_steps, device)
            rewards.append(reward)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        grad_estimate = torch.zeros_like(baseline_params)
        for i in range(population_size):
            grad_estimate += rewards_normalized[i] * noise_list[i]
        grad_estimate /= population_size

        # 進化戦略による更新
        baseline_params = baseline_params + learning_rate * grad_estimate
        set_flat_params(model.controller, baseline_params)
        
        avg_reward = rewards_tensor.mean().item()
        grad_norm = torch.norm(grad_estimate).item()
        
        print(f"Iteration {iteration+1}/{num_iterations}, Average Reward: {avg_reward}")

        # TensorBoard ログ記録
        writer.add_scalar("Reward/Average", avg_reward, iteration)
        writer.add_scalar("Reward/Best", best_reward, iteration)
        writer.add_scalar("Gradient/Norm", grad_norm, iteration)
        writer.add_scalar("Sigma", sigma, iteration)

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = baseline_params.clone()
            print(f"New best model found with reward {best_reward}, saving checkpoint...")
            torch.save(best_params, os.path.join(save_dir, "best_controller.pth"))

    torch.save(baseline_params, os.path.join(save_dir, "final_controller.pth"))
    print("Final model saved.")
    
    writer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
