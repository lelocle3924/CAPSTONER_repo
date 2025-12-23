# file: main.py ver 3

import argparse
import numpy as np
import os
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from core import LightLogger
from config import PPOConfig, PathConfig
from ppo.rvrpenv import RVRPEnvironment
from core.visualizer import RouteVisualizer

ppo_cfg = PPOConfig()
path_cfg = PathConfig()

parser = argparse.ArgumentParser(description="VRPTW Optimization Engine")
parser.add_argument("-m", "--mode", type=str, choices=["train", "test"], default="train", help="Run mode")
parser.add_argument("-p", "--model_path", type=str, default=None, help="Path to load model")
args = parser.parse_args()

logger = LightLogger(name="Main")

def mask_fn(env):
    return env.valid_action_mask()

def make_env(is_test_mode=False):
    def _init():
        # [UPDATED] Use centralized paths
        env = RVRPEnvironment(
            order_csv_path=path_cfg.ORDER_PATH,
            truck_csv_path=path_cfg.TRUCK_PATH,
            is_test_mode=is_test_mode
        )
        env = ActionMasker(env, mask_fn)
        return env
    return _init

def train_ppo():
    # Directories
    os.makedirs(ppo_cfg.model_save_path, exist_ok=True)
    os.makedirs(ppo_cfg.tensorboard_log, exist_ok=True)

    # Env
    logger.info(f"Initializing {ppo_cfg.num_envs} parallel environments...")
    train_env = DummyVecEnv([make_env(is_test_mode=False) for _ in range(ppo_cfg.num_envs)])
    train_env = VecMonitor(train_env, ppo_cfg.monitor_path)

    # Model
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Resuming training from {args.model_path}...")
        model = MaskablePPO.load(args.model_path, env=train_env, device=ppo_cfg.device)
    else:
        logger.info("Initializing NEW MaskablePPO model...")
        model = MaskablePPO(
            "MultiInputPolicy", 
            train_env, 
            n_steps=ppo_cfg.n_steps, 
            batch_size=ppo_cfg.batch_size,
            n_epochs=ppo_cfg.n_epochs,
            learning_rate=ppo_cfg.learning_rate,
            verbose=1, 
            ent_coef=0.01,
            device=ppo_cfg.device,
            tensorboard_log=ppo_cfg.tensorboard_log
        )

    # Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=ppo_cfg.save_freq, 
        save_path=ppo_cfg.model_save_path, 
        name_prefix="ppo_v2_nightly"
    )

    logger.info(f"STARTING TRAINING: Target {ppo_cfg.total_timesteps} steps.")
    model.learn(
        total_timesteps=ppo_cfg.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    final_path = os.path.join(ppo_cfg.model_save_path, "final_model_nightly.zip")
    model.save(final_path)
    logger.info(f"Training Complete. Model saved to {final_path}")

def evaluate_model():
    if not args.model_path or not os.path.exists(args.model_path):
        logger.error("Please provide a valid model path using -p")
        return

    logger.info(f"Evaluating model: {args.model_path}")
    visualizer = RouteVisualizer()
    env = make_env(is_test_mode=True)()
    
    model = MaskablePPO.load(args.model_path, device=ppo_cfg.device)
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        # action_masks is handled by SB3's predict if env is wrapped with ActionMasker? 
        # Wait, for predict we might need to pass mask manually if not using VecEnv in a specific way.
        # But ActionMasker puts mask in info/observation usually. 
        # Correct way for single env unwrapped prediction:
        from sb3_contrib.common.maskable.utils import get_action_masks
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if done:
            logger.info(f"Evaluation Finished. Best Cost: {info.get('best_cost')}")
            # Visualize
            real_env = env.unwrapped
            best_sol = real_env.best_solution
            visualizer.visualize_solution(best_sol, real_env.problem_data, filename="nightly_eval_result.html")

if __name__ == "__main__":
    if args.mode == "train":
        train_ppo()
    else:
        evaluate_model()