# file: config.py ver 3
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

@dataclass
class PathConfig:
    """Centralized File Paths"""
    # Inputs
    ORDER_PATH: str = "inputs/CleanData/Split_TransportOrder_2524.csv" # Hoặc Split_TransportOrder_2524.csv
    TRUCK_PATH: str = "inputs/MasterData/TruckMaster.csv"
    
    # Outputs / Cache
    DISTANCE_TIME_PATH: str = "inputs/DistTimeMatrix"
    RESULTS_DIR: str = "results"
    LOGS_DIR: str = "logs"
    MODELS_DIR: str = "models"

@dataclass
class PPOConfig:
    """Hyperparameters for PPO Agent & Training Loop"""
    # --- TRAINING PARAMETERS (UPDATED FOR OVERNIGHT RUN) ---
    train_seed: int = 2025
    num_envs: int = 4             # Parallel environments
    n_steps: int = 2048           # Steps per env per update
    batch_size: int = 64          # Minibatch size
    n_epochs: int = 10            # Epochs per update
    learning_rate: float = 3e-4   # Stable LR
    
    # Total Timesteps: 1 Million for overnight
    current_trained_timesteps: int = 0
    total_timesteps: int = 50_000 
    
    save_freq: int = 5000         # Save model every 5k steps (Safety)
    
    # --- REWARD FUNCTION (TUNED) ---
    # Tăng trọng số Utilization để ép Agent lấp đầy xe
    reward_cost_scale: float = 4.0
    reward_util_lambda: float = 0.2  # UPDATED: Was 0.05
    stop_threshold: float = 1000     # Early stopping patience
    
    # --- DEVICE ---
    device: str = "cpu" # or "cuda" if available
    
    # --- LOGGING PATHS ---
    # Auto-generated based on logic
    def __post_init__(self):
        now_str = datetime.now().strftime("%m%d_%H%M")
        self.run_name = f"{now_str}_{self.total_timesteps}steps"
        self.tensorboard_log = os.path.join(PathConfig.LOGS_DIR, "tensorboard_runs")
        self.model_save_path = os.path.join(PathConfig.MODELS_DIR, self.run_name)
        self.monitor_path = os.path.join(self.model_save_path, "monitor.csv")

@dataclass
class ALNSConfig:
    """Hyperparameters for the Heuristic Engine"""
    seed: int = 520
    num_iterations: int = 100 # iterations per PPO step
    
    # Operator counts (Will be auto-filled by environment registration)
    num_destroy: int = 0 
    num_repair: int = 0
    
    # Internal ALNS logic
    roulette_wheel_scores: list[float] = field(default_factory=lambda: [25, 5, 1, 0])
    roulette_wheel_decay: float = 0.9

    def get_alns_params_dict(self):
        return asdict(self)