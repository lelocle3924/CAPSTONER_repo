import os
from dataclasses import asdict, dataclass, field
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

@dataclass
class PPOConfig:
    # GENERATOR CONFIG
    generator_type: str = "R"  # Generator type ["R", "C"]
    
    dimension: int = 151  # SỬA Ở ĐÂY NÈ
    capacity: int = 64  # Vehicle capacity
    num_vehicles: int = dimension  # Number of vehicles
    max_travel_time: int = 50  # THỜI LƯỢNG TỐI ĐA CHO 1 ROUTE LÀ BAO NHIÊU ----------------> CHÍNH LÀ CHỖ CONSTRAINTS MUST NOT BE MORE THAN 10 HOUR ROUTES
    depot = ''
    demand = ''
    time_windows = ''
    service_times = ''
    travel_times = ''
    node_coords = ''
    edge_weights = ''
    
    max_demand: int = 16  # Maximum demand    
    min_window_width: int = 5  # Minimum time window width
    max_window_width: int = 15  # Maximum time window width
    

    # TRAIN CONFIG
    current_trained_timesteps = 12800
    num_envs: int = 8
    n_steps: int = 2000
    
    total_timesteps: int = current_trained_timesteps + num_envs*n_steps  # SỬA Ở ĐÂY NÈ

    train_seed: int = 2025  # Training random seed
    save_freq: int = 1600  # Save frequency
    batch_size: int = 40  # Training batch size
    n_epochs: int = 100  # Number of epochs per update
    learning_rate: float = 1e-3  # Initial learning rate
    progress_bar: bool = True  # Whether to show progress bar
    device: str = "cpu"  # 
    training_pool_size: int = 128  # Training instance pool size
    training_refresh_threshold: int = (128 * 32)  # Training instance pool refresh threshold
    training_sampling_memory: int = 128  # Training instance pool sampling memory
    training_chunk_size: int = 32  # Training instance pool batch size
    training_max_workers: int = 8  # Training instance pool max workers
    training_no_refresh: bool = (False)  # Whether training instance pool should not refresh
    
    # TEST CONFIG
    n_eval_envs: int = 1  
    eval_seed: int = 123 
    eval_freq: int = 2000 
    eval_pool_size: int = 100 
    eval_sampling_memory: int = 100  
    eval_chunk_size: int = 20 
    eval_max_workers: int = 2 
    eval_no_refresh: bool = True 

    test_seed: int = 369
    test_instances: int = 10 
    stop_threshold: float = 1000 #     SỐ LẦN AGENT RA QUYẾT ĐỊNH STOP TRƯỚC KHI KẾT THÚC EPISODE
    reward_util_lambda = 0.05
    reward_cost_scale = 4
    # OTHER CONFIG
    num_flag: int = 151 #dimension
    tensorboard_log: str = f"./logs/dim{num_flag}cap{capacity}/log{total_timesteps}/"
    model_save_path: str = (f"./models/model{num_flag}_{generator_type}/capacity_{capacity}")
    log_dir = tensorboard_log
    monitor_filename = f"monitor.csv"
    monitor_path = os.path.join(log_dir, monitor_filename)


@dataclass
class ALNSConfig:
    seed: int = 520  # Algorithm random seed
    num_iterations: int = 125  # Number of iterations
    num_destroy: int = 5  # Number of destroy operators
    num_repair: int = 3  # Number of repair operators
    roulette_wheel_scores: list[float] = field(
        default_factory=lambda: [25, 5, 1, 0]
    )  # Roulette wheel score list
    roulette_wheel_decay: float = 0.9  # Roulette wheel decay coefficient
    autofit_start_gap: float = 0.05  # Initially only accept solutions within 0.* coefficient of optimal solution
    autofit_end_gap: float = 0  # Finally accept solutions within 0.* coefficient of optimal solution (usually 0)

    def get_alns_params_dict(self):
        return asdict(self)
