
import argparse
import time
import numpy as np
import os
import torch
import pickle
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from core import LightLogger, VRPTWGeneratorC, VRPTWGeneratorR
from config import PPOConfig
from ppo.vrptwenv import VRPTWEnvironment
from ppo.instance_pool import InstancePool
from core import format_vrptw_routes

ppo_config = PPOConfig()
parser = argparse.ArgumentParser(description="VRPTW Solver")
parser.add_argument("-a", "--algorithm", type=str, default="ppo", help="Algorithm")
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    choices=["train", "test"],
    default="train",
    help="Run mode",
)
parser.add_argument(
    "-p",
    "--model_path",
    type=str,
    default=os.path.join(ppo_config.model_save_path, f"final_model_{ppo_config.total_timesteps}.zip"),
    help="Model loading path for testing or continuing training.",
)
args = parser.parse_args()

logger = LightLogger(name="Main")

if ppo_config.generator_type == "R":
    generator = VRPTWGeneratorR(dimension=ppo_config.dimension, capacity=ppo_config.capacity, max_demand=ppo_config.max_demand, num_vehicles=ppo_config.num_vehicles, min_window_width=ppo_config.min_window_width, max_window_width=ppo_config.max_window_width, max_travel_time=ppo_config.max_travel_time)
elif ppo_config.generator_type == "C":
    generator = VRPTWGeneratorC(dimension=ppo_config.dimension, capacity=ppo_config.capacity, max_demand=ppo_config.max_demand, num_vehicles=ppo_config.num_vehicles, min_window_width=ppo_config.min_window_width, max_window_width=ppo_config.max_window_width, max_travel_time=ppo_config.max_travel_time)

def make_env(in_pool, is_test_mode=False):
    def _init():
        env = VRPTWEnvironment(instance_pool=in_pool, is_test_mode=is_test_mode)
        return env
    return _init

def create_subproc_env(n_envs, in_pool):
    if __name__ == '__main__':
        return SubprocVecEnv([make_env(in_pool) for _ in range(n_envs)])
    return None

def create_dummy_env(n_envs, in_pool, is_test_mode=False):
    return DummyVecEnv([make_env(in_pool, is_test_mode=is_test_mode) for _ in range(n_envs)])

def train_ppo(
    n_envs: int, n_steps: int, batch_size: int, n_epochs: int, learning_rate: float,
    total_timesteps: int, progress_bar: bool, save_path: str, model_path: str
):
    instance_pool = InstancePool(generator, pool_size=ppo_config.training_pool_size, refresh_threshold=ppo_config.training_refresh_threshold, sampling_memory=ppo_config.training_sampling_memory, chunk_size=ppo_config.training_chunk_size, max_workers=ppo_config.training_max_workers, seed=ppo_config.train_seed)
    
    os.makedirs(save_path, exist_ok=True)
    
    
    os.makedirs(ppo_config.log_dir, exist_ok=True)
    
    # TẠO TRAIN_ENV BẰNG SUBPROC_VEC_ENV, THỨ ĐƯỢC TẠO BẰNG MAKE_ENV, THỨ ĐƯỢC TẠO BẰNG VRPTW_ENVIRONMENT
    train_env = create_subproc_env(n_envs, instance_pool)
    if train_env is None:
        logger.warning("SubprocVecEnv failed to initialize, falling back to DummyVecEnv.")
        train_env = create_dummy_env(n_envs, instance_pool)
    
    
    train_env = VecMonitor(train_env, ppo_config.monitor_path)

    #LOAD EXISTING MODEL IF TRUE
    if os.path.exists(args.model_path) and args.mode == 'train':
        logger.info(f"Loading model from {args.model_path} to continue training...")
        model = PPO.load(args.model_path, env=train_env, device=ppo_config.device)
        timesteps_trained_before = model.num_timesteps
        logger.info(f"Model has already been trained for {timesteps_trained_before} timesteps.")
    else:
        logger.info("Creating a new model...")
        timesteps_trained_before = 0
        def linear_schedule(initial_value: float):
            def func(progress_remaining: float) -> float:
                return progress_remaining * initial_value
            return func

        #NHÉT TRAIN_ENV VÀO PPO CHO NÓ TỰ XÀI, KHÔNG CAN THIỆP KHI NÀO TÍNH REWARD, KHI NÀO STEP, BLA BLA
        model = PPO(
            "MultiInputPolicy", train_env, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
            learning_rate=linear_schedule(learning_rate), verbose=1, ent_coef=0.001,
            policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]), activation_fn=torch.nn.ReLU),
            device=ppo_config.device,
        )
    
    checkpoint_callback = CheckpointCallback(save_freq=ppo_config.save_freq, save_path=save_path, name_prefix="ppo4vrptw")
    
    # Tính toán số timesteps cần train thêm
    additional_timesteps = total_timesteps - timesteps_trained_before
    if additional_timesteps <= 0:
        logger.warning(f"Total timesteps in config ({total_timesteps}) is not greater than the timesteps already trained ({timesteps_trained_before}). No new training will occur.")
        return

    try:
        model.learn(
            total_timesteps=additional_timesteps,
            callback=checkpoint_callback,
            progress_bar=progress_bar,
            reset_num_timesteps=False
        )
    finally:
        train_env.close()

    final_model_name = f"final_model_{total_timesteps}.zip"
    final_model_path = os.path.join(save_path, final_model_name)
    model.save(final_model_path)
    logger.info(f"The model has been saved to {final_model_path}")


def evaluate_model(model_path):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    ppo_costs = []
    ppo_behavior_logs = []

    test_pool = InstancePool(
        generator, pool_size=ppo_config.test_instances, sampling_memory=ppo_config.test_instances,
        max_workers=1, chunk_size=1, seed=ppo_config.test_seed, sequential=True
    )
    
    env = DummyVecEnv([make_env(test_pool, is_test_mode=True)])

    model = PPO.load(model_path, env=env, device=ppo_config.device)
    
    for i in range(ppo_config.test_instances):
        obs = env.reset()
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True) # _states đó insignificant, vì sẽ phải step trong ENV mới biết được
            obs, reward, done, info = env.step(action)
        
        info_dict = info[0]
                
        logger.info(f"--- Finished PPO on instance {i} ---")
        logger.info(f"PPO Result Costs: {info_dict['best_cost']}")
        
        instance_data = info_dict['instance_data']
        best_solution = info_dict['best_solution']
        behavior_log = info_dict['behavior_log']

        if not behavior_log:
             logger.warning(f"Instance {i}: Behavior log is empty!")

        ppo_costs.append(best_solution.objective())

        with open(os.path.join(results_dir, f"instance_{i}.pkl"), "wb") as f:
            pickle.dump(instance_data, f)
        with open(os.path.join(results_dir, f"ppo_solution_{i}.pkl"), "wb") as f:
            pickle.dump(best_solution, f)
        
        df_behavior = pd.DataFrame(behavior_log)
        df_behavior['instance_id'] = i
        ppo_behavior_logs.append(df_behavior)

    pd.DataFrame(ppo_costs, columns=['cost']).to_csv(os.path.join(results_dir, "ppo_costs.csv"), index=False)
    if ppo_behavior_logs:
        all_logs = pd.concat(ppo_behavior_logs, ignore_index=True)
        if not all_logs.empty:
            all_logs.to_csv(os.path.join(results_dir, "ppo_behavior_log.csv"), index=False)
    logger.info(f"Results saved to '{results_dir}' directory.")
    env.close()
    

if __name__ == "__main__":
    if args.mode == "train":
        train_ppo(
            n_envs=ppo_config.num_envs, n_steps=ppo_config.n_steps, batch_size=ppo_config.batch_size,
            n_epochs=ppo_config.n_epochs, learning_rate=ppo_config.learning_rate,
            total_timesteps=ppo_config.total_timesteps, progress_bar=ppo_config.progress_bar,
            save_path=ppo_config.model_save_path, model_path=args.model_path,
        )
    elif args.mode == "test":
        if not os.path.exists(args.model_path):
            logger.error(f"Model file not found at: {args.model_path}")
            logger.error("Please train a model first using '--mode train'")
        else:
            evaluate_model(args.model_path)