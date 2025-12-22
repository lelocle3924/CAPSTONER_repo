# file: main.py

import argparse
import time
import numpy as np
import os
import torch
import pickle
import pandas as pd

# Tắt cảnh báo oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- NEW IMPORTS FOR MASKABLE PPO ---
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from core import LightLogger
from config import PPOConfig
# Lưu ý: Import class RVRPEnvironment từ file mới
from ppo.rvrpenv import RVRPEnvironment
from ppo.instance_pool import InstancePool
from core.visualizer import RouteVisualizer # New Visualizer

# ... (Giữ nguyên Parser và Config) ...
ppo_config = PPOConfig()
parser = argparse.ArgumentParser(...) 
# ...
args = parser.parse_args()
logger = LightLogger(name="Main")

# --- Helper functions ---

def mask_fn(env: gym.Env) -> np.ndarray:
    """Wrapper function để lấy mask từ env"""
    return env.valid_action_mask()

def make_env(is_test_mode=False):
    """
    Factory function tạo Env đã được bọc Masker.
    Lưu ý: Real Data Loader dùng path cứng trong Config hoặc Env, 
    ở đây hardcode path input cho khớp với yêu cầu.
    """
    def _init():
        env = RVRPEnvironment(
            order_csv_path="inputs/CleanData/Split_TransportOrder_1day.csv",
            truck_csv_path="inputs/MasterData/TruckMaster.csv",
            is_test_mode=is_test_mode
        )
        # Wrap env với ActionMasker
        env = ActionMasker(env, mask_fn)
        return env
    return _init

# --- TRAIN FUNCTION (Updated for MaskablePPO) ---

def train_ppo(n_envs, n_steps, total_timesteps, save_path, model_path, **kwargs):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(ppo_config.log_dir, exist_ok=True)

    # Create Vectorized Env
    # Lưu ý: MaskablePPO hỗ trợ VecEnv, nhưng ActionMasker phải bọc từng env con bên trong Dummy/Subproc
    train_env = DummyVecEnv([make_env(is_test_mode=False) for _ in range(n_envs)])
    train_env = VecMonitor(train_env, ppo_config.monitor_path)

    # Load or Create Model
    if os.path.exists(args.model_path) and args.mode == 'train':
        logger.info(f"Loading MaskablePPO model from {args.model_path}...")
        model = MaskablePPO.load(args.model_path, env=train_env, device=ppo_config.device)
    else:
        logger.info("Creating new MaskablePPO model...")
        model = MaskablePPO(
            "MultiInputPolicy", 
            train_env, 
            n_steps=n_steps, 
            verbose=1, 
            ent_coef=0.01, # Tăng entropy để explore tốt hơn lúc đầu
            device=ppo_config.device,
            tensorboard_log=ppo_config.tensorboard_log
        )

    checkpoint_callback = CheckpointCallback(save_freq=ppo_config.save_freq, save_path=save_path, name_prefix="ppo_maskable")

    logger.info("Starting Training with Action Masking...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=kwargs.get('progress_bar', True),
        reset_num_timesteps=False
    )
    
    final_path = os.path.join(save_path, "final_model_maskable.zip")
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")

# --- EVALUATE FUNCTION (Updated with Visualization) ---

def evaluate_model(model_path):
    logger.info("Starting Evaluation...")
    visualizer = RouteVisualizer()
    
    # Create Test Env (Single instance)
    env = make_env(is_test_mode=True)() # Unwrapped for manual stepping or wrapped?
    # MaskablePPO requires env to be wrapped if we use predict(deterministic=True) potentially
    # Nhưng để an toàn ta cứ wrap
    
    # Load Model
    model = MaskablePPO.load(model_path, device=ppo_config.device)
    
    obs, _ = env.reset()
    done = False
    
    logger.info("Running optimization loop...")
    while not done:
        # Retrieve valid action mask from env
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if done:
            best_cost = info.get('best_cost', 'N/A')
            logger.info(f"Episode Finished. Best Cost: {best_cost}")
            
            # --- VISUALIZATION STEP ---
            # Lấy solution object từ env (cần truy cập biến nội bộ env.unwrapped)
            if hasattr(env, 'unwrapped'):
                real_env = env.unwrapped
            else:
                real_env = env
                
            best_sol = real_env.best_solution
            problem_data = real_env.problem_data
            
            # Export Map
            visualizer.visualize_solution(
                best_sol, 
                problem_data, 
                filename=f"final_route_cost_{int(best_sol.objective())}.html"
            )
            
            # Export CSV Report
            _export_csv_report(best_sol, problem_data)

def _export_csv_report(solution, data):
    """Xuất file Excel/CSV theo yêu cầu Team A để so sánh"""
    rows = []
    for r_idx, route in enumerate(solution.routes):
        v_type = route.vehicle_type.name
        
        current_time = 0 # Cần logic tính time thực tế giống visualizer
        # Simplified report
        for seq, node_idx in enumerate(route.node_sequence):
            rows.append({
                "VehicleID": f"V_{r_idx}_{v_type}",
                "VehicleType": v_type,
                "StopSequence": seq + 1,
                "LocationID": data.node_ids[node_idx],
                "LoadKG": data.demands_kg[node_idx],
                "ServiceTime": data.service_times[node_idx]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv("results/final_schedule.csv", index=False)
    print("  > Schedule exported to results/final_schedule.csv")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if args.mode == "train":
        train_ppo(
            n_envs=ppo_config.num_envs, 
            n_steps=ppo_config.n_steps, 
            total_timesteps=ppo_config.total_timesteps,
            save_path=ppo_config.model_save_path, 
            model_path=args.model_path,
            progress_bar=ppo_config.progress_bar
        )
    elif args.mode == "test":
        if not os.path.exists(args.model_path):
            logger.error(f"Model not found: {args.model_path}")
        else:
            evaluate_model(args.model_path)