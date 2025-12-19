import numpy as np
import pandas as pd
import pickle
import os

from core import LightLogger, VRPTWGeneratorR, VRPTWGeneratorC
from config import PPOConfig, ALNSConfig
from base_alns.vrp import runner

# --- KHỞI TẠO ---
ppo_config = PPOConfig()
alns_config = ALNSConfig()
logger = LightLogger(name="Baseline")
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# --- TẠO GENERATOR (giống hệt main.py) ---
if ppo_config.generator_type == "R":
    generator = VRPTWGeneratorR(
        dimension=ppo_config.dimension, capacity=ppo_config.capacity, max_demand=ppo_config.max_demand,
        num_vehicles=ppo_config.num_vehicles, min_window_width=ppo_config.min_window_width,
        max_window_width=ppo_config.max_window_width, max_travel_time=ppo_config.max_travel_time,
    )
elif ppo_config.generator_type == "C":
    generator = VRPTWGeneratorC(
        dimension=ppo_config.dimension, capacity=ppo_config.capacity, max_demand=ppo_config.max_demand,
        num_vehicles=ppo_config.num_vehicles, min_window_width=ppo_config.min_window_width,
        max_window_width=ppo_config.max_window_width, max_travel_time=ppo_config.max_travel_time,
    )

# --- CHẠY VÀ LƯU KẾT QUẢ ---
logger.info(f"--- Running Baseline ALNS on {ppo_config.test_instances} instances ---")

all_costs = []
# Sử dụng cùng seed và thứ tự với PPO test để đảm bảo công bằng
for i in range(ppo_config.test_instances):
    seed = ppo_config.test_seed + i
    instance_data = generator.generate(seed=seed).get_data()
    
    logger.info(f"\n--- Solving instance {i} (seed={seed}) ---")
    
    result = runner.alns4vrptw(
        data=instance_data,
        params=alns_config.get_alns_params_dict(),
        logger=logger
    )
    
    best_solution = result.best_state
    best_cost = best_solution.objective()
    all_costs.append(best_cost)
    
    # Lưu solution của baseline
    with open(os.path.join(results_dir, f"baseline_solution_{i}.pkl"), "wb") as f:
        pickle.dump(best_solution, f)

# Lưu chi phí của baseline
pd.DataFrame(all_costs, columns=['cost']).to_csv(os.path.join(results_dir, "baseline_costs.csv"), index=False)

logger.info("\n--- Baseline Run Complete ---")
logger.info(f"Results saved to '{results_dir}' directory.")
logger.info(f"Average cost for Baseline ALNS: {np.mean(all_costs):.4f}")