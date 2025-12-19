import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

# Import from single source of truth
from core.data_structures import ProblemData, PPOState, RvrpState, Route
from core.real_data_loader import RealDataLoader
from config import PPOConfig, ALNSConfig
import alns.vrp4ppo as vrp
from alns.alns4ppo import ALNS4PPO

ppo_config = PPOConfig()
alns_config = ALNSConfig()

#ver 4: tính utilization natively trong RvrpState, đổi tên class thành RVRPEnvironment, thay vì VRPTWEnvironment
# 4.1: register destroy severity cho các destroy operator
# 4.2: 

class RVRPEnvironment(gym.Env):
    def __init__(self, 
                 order_csv_path: str, 
                 dist_csv_path: str, 
                 time_csv_path: str, 
                 is_test_mode: bool = False):
        super().__init__()
        
        # 1. Load Data
        self.loader = RealDataLoader()
        self.problem_data: ProblemData = self.loader.load_day_data(
            order_csv_path, dist_csv_path, time_csv_path
        )
        
        self.is_test_mode = is_test_mode
        self.alns = ALNS4PPO()
        
        # 2. Spaces
        self.d_op_num = alns_config.num_destroy
        self.r_op_num = alns_config.num_repair
        
        self.action_space = spaces.MultiDiscrete([self.d_op_num, self.r_op_num, 2, 2])
        
        obs_size = PPOState.get_observation_size(self.d_op_num, self.r_op_num)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # 3. Tracking
        self.iters = 0
        self.stop_counter = 0
        self.stagnation_counter = 0
        self.destroy_usage = np.zeros(self.d_op_num, dtype=np.float32)
        self.repair_usage = np.zeros(self.r_op_num, dtype=np.float32)
        
        # State Objects (RvrpState)
        self.current_solution: RvrpState = None
        self.best_solution: RvrpState = None
        self.ppo_state: PPOState = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.iters = 0
        self.stop_counter = 0
        self.stagnation_counter = 0
        self.destroy_usage.fill(0)
        self.repair_usage.fill(0)
        
        self.alns.reset_opt()
        self._register_operators()

        # Generate Initial Solution (Must return RvrpState)
        self.init_solution = vrp.initial.nearest_neighbor_tw(self.problem_data)
        
        self.current_solution = self.init_solution.copy()
        self.pre_solution = self.init_solution.copy()
        self.best_solution = self.init_solution.copy()
        
        # Init cost for normalization
        self.initial_cost = self.init_solution.objective()
        if self.initial_cost == 0: self.initial_cost = 1.0

        self.ppo_state = self._extract_features()
        return self.ppo_state.to_array(), {}

    def step(self, action):
        d_idx, r_idx, accept, stop = action
        self.iters += 1
        
        self.destroy_usage[d_idx] += 1
        self.repair_usage[r_idx] += 1
        
        # ALNS Iteration
        self.pre_solution, self.current_solution = self.alns.iterate(
            self.current_solution, self.pre_solution, d_idx, r_idx, accept, 
            data=self.problem_data
        )

        reward = self._calculate_reward(accept)
        
        if self.current_solution.objective() < self.best_solution.objective():
            self.best_solution = self.current_solution.copy()
            self.stagnation_counter = 0
            reward += 1.0 
        else:
            self.stagnation_counter += 1

        self.ppo_state = self._extract_features()
        
        terminated = (stop == 1 and self.stop_counter >= 2) or (self.iters >= alns_config.num_iterations)
        truncated = False
        
        info = {}
        if terminated:
            info = {
                "best_cost": self.best_solution.objective(),
                "num_vehicles": len(self.best_solution.routes),
                "mean_utilization": self.ppo_state.mean_cap_utilization
            }

        return self.ppo_state.to_array(), reward, terminated, truncated, info

    def _extract_features(self) -> PPOState:
        """
        Calculates the 21+ features for the PPO neural network.
        Logic is now delegated to RvrpState properties.
        """
        # 1. Solution Statistics (Delegated)
        sol = self.current_solution
        num_routes = len(sol.routes)
        
        # 2. Instance Stats (Proxy)
        avg_dist_to_depot = np.mean(self.problem_data.dist_matrix[0, 1:])
        density_proxy = self.problem_data.num_nodes / (avg_dist_to_depot + 1.0)

        # 3. Construct PPOState
        return PPOState(
            search_progress = self.iters / alns_config.num_iterations,
            stagnation_norm = min(1.0, self.stagnation_counter / 100.0),
            best_cost_norm = self.best_solution.objective() / self.initial_cost,
            current_cost_norm = sol.objective() / self.initial_cost,
            improvement_history = 0.0, 
            
            demands_mean = np.mean(self.problem_data.demands_kg),
            demands_std = np.std(self.problem_data.demands_kg),
            tw_width_mean = np.mean(self.problem_data.time_windows[:, 1] - self.problem_data.time_windows[:, 0]),
            tw_tightness = 0.0, 
            spatial_density = density_proxy,
            
            # [UPDATED] Get directly from RvrpState properties
            min_cap_utilization = sol.min_capacity_utilization,
            mean_cap_utilization = sol.mean_capacity_utilization,
            max_cap_utilization = sol.max_capacity_utilization,
            
            num_routes_norm = num_routes / self.problem_data.num_nodes, 
            num_unassigned_norm = len(sol.unassigned) / self.problem_data.num_nodes,
            
            destroy_probs = self.destroy_usage / (np.sum(self.destroy_usage) + 1e-6),
            repair_probs = self.repair_usage / (np.sum(self.repair_usage) + 1e-6)
        )

    def _calculate_reward(self, accept_action):
        """
        Reward Function Strategy:
        1. Primary Goal: Cost Reduction (Total Distance + Fixed Cost)
        2. Secondary Goal: Compactness (Capacity Utilization)
        """
        # 1. Primary Component: Cost Improvement
        prev_cost = self.pre_solution.objective()
        curr_cost = self.current_solution.objective()
        
        # Calculate relative improvement
        # Avoid division by zero
        base_cost = prev_cost if prev_cost > 1e-6 else 1.0
        cost_delta_pct = (prev_cost - curr_cost) / base_cost
        
        reward_cost = cost_delta_pct * ppo_config.reward_cost_scale
        
        # 2. Secondary Component: Utilization Improvement (Bonus)
        # RvrpState has the property 'mean_capacity_utilization'
        prev_util = self.pre_solution.mean_capacity_utilization
        curr_util = self.current_solution.mean_capacity_utilization
        
        util_delta = curr_util - prev_util
        
        # Apply Lambda coefficient
        reward_util = util_delta * ppo_config.reward_util_lambda
        
        # 3. Total Reward
        total_reward = reward_cost + reward_util
        
        # Optional: Penalty for invalid moves (if needed later)
        # But ALNS handles feasibility internally, so no penalty needed here usually.
        
        return total_reward


    def _register_operators(self):
        # DESTROY OPTS
        # 1. Random Removal (3 levels)
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data, ratio=0.1)) # Light
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data, ratio=0.25)) # Medium
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data, ratio=0.4)) # Heavy

        # 2. Worst Removal (2 levels) - Cái này quan trọng để phá cục bộ
        self.alns.add_destroy_operator(vrp.create_worst_removal_operator(self.problem_data, ratio=0.1))
        self.alns.add_destroy_operator(vrp.create_worst_removal_operator(self.problem_data, ratio=0.3))

        # 3. Route Removal (Phá vỡ cấu trúc xe)
        self.alns.add_destroy_operator(vrp.create_random_route_removal_operator(ratio=0.1)) # Xóa 10% số xe

        # REPAIR OPTS
        self.alns.add_repair_operator(vrp.create_greedy_repair_operator(self.problem_data))
        self.alns.add_repair_operator(vrp.create_regret_repair_operator(self.problem_data))