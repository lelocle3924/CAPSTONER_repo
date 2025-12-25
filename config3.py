# file: config.py
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

@dataclass
class PathConfig:
    ORDER_PATH: str = "inputs/CleanData/Split_TransportOrder_allabove1_2524.csv"
    TRUCK_PATH: str = "inputs/MasterData/TruckMaster.csv"
    DISTANCE_TIME_PATH: str = "inputs/DistTimeMatrix"
    RESULTS_DIR: str = "results"
    LOGS_DIR: str = "logs"
    MODELS_DIR: str = "models"
    MODEL_PATH: str = "models/VRPTW_1225_1239/vrp_model_65000_steps.zip"

@dataclass
class PPOConfig:
    train_seed: int = 2025
    # [TRIAL SETTINGS] 
    # Use 4-8 for Colab. If CPU hits 100% and GPU is low, decrease this.
    num_envs: int = 4             
    
    # [GPU THROUGHPUT]
    # n_steps * num_envs = Total samples per update. 
    # 2048 * 4 = 8192 samples per update.
    n_steps: int = 2048           
    batch_size: int = 256         # Increased from 64 for better GPU utilization
    n_epochs: int = 10            
    learning_rate: float = 3e-4   
    ent_coef: float = 0.01        
    
    # [TIMESTEPS]
    current_trained_timesteps: int = 65000
    total_timesteps: int = 500_000 # 1M for overnight run
    save_freq: int = 5000         # Save every 10k steps
    
    reward_cost_scale: float = 5.0  # Increased weight on cost
    reward_util_lambda: float = 0.5 # Stronger push for full trucks
    stop_threshold: int = 50        # Stop episode after 50 steps of no improvement

    base_step_penalty: float = 0.005    # Phạt nhẹ mỗi step RL
    operator_time_limit: float = 2.0    # Ngưỡng thời gian (giây) cho 1 cặp D+R. Vượt ngưỡng này sẽ bị phạt thêm.
    operator_penalty_scale: float = 0.01
    
    device: str = "cuda"            # Force CUDA

    def __post_init__(self):
        now_str = datetime.now().strftime("%m%d_%H%M")
        self.run_name = f"VRPTW_{now_str}"
        self.tensorboard_log = os.path.join(PathConfig.LOGS_DIR, "tb")
        self.model_save_path = os.path.join(PathConfig.MODELS_DIR, self.run_name)
        self.monitor_path = os.path.join(self.model_save_path, "monitor.csv")

@dataclass
class ALNSConfig:
    num_iterations: int = 100      # 100 ALNS iterations per 1 RL Step