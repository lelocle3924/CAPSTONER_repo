import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

from core.data_structures import ProblemData, PPOState
from core.real_data_loader import RealDataLoader
from config import PPOConfig, ALNSConfig
import alns.vrp as vrp
from alns.alns4ppo import ALNS4PPO

ppo_config = PPOConfig()
alns_config = ALNSConfig()

#ver 2, đã define state class ở file data_structures.py, obs space đã fixed size, không phụ thuộc dimension
# đã có function extract_features để tính các features, sau đó nạp vào PPOState và trả lại

class VRPTWEnvironment(gym.Env):
    """
    Gymnasium Environment for VRPTW optimization using PPO-ALNS.
    Connects the Real Data, ALNS Algorithm, and PPO Agent.
    """
    def __init__(self, 
                 order_csv_path: str, 
                 dist_csv_path: str, 
                 time_csv_path: str, 
                 is_test_mode: bool = False):
        super().__init__()
        
        # 1. Load Real Data
        self.loader = RealDataLoader()
        self.problem_data: ProblemData = self.loader.load_day_data(
            order_csv_path, dist_csv_path, time_csv_path
        )
        
        self.is_test_mode = is_test_mode
        self.alns = ALNS4PPO()
        
        # 2. Config Dimensions & Operators
        self.d_op_num = alns_config.num_destroy
        self.r_op_num = alns_config.num_repair
        
        # 3. Define Action Space
        # [Destroy Op Index, Repair Op Index, Accept (0/1), Stop (0/1)]
        self.action_space = spaces.MultiDiscrete([self.d_op_num, self.r_op_num, 2, 2])

        # 4. Define Observation Space (Fixed-size Box)
        obs_size = PPOState.get_observation_size(self.d_op_num, self.r_op_num)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Internal tracking variables
        self.iters = 0
        self.stop_counter = 0
        self.stagnation_counter = 0
        self.stagnation_threshold = 100
        self.destroy_usage = np.zeros(self.d_op_num, dtype=np.float32)
        self.repair_usage = np.zeros(self.r_op_num, dtype=np.float32)
        self.current_solution = None
        self.best_solution = None
        self.ppo_state = None

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        
        # Reset counters
        self.iters = 0
        self.stop_counter = 0
        self.destroy_usage.fill(0)
        self.repair_usage.fill(0)
        
        # Reset ALNS operators
        self.alns.reset_opt()
        
        # Register operators (Assuming vrp.create_* functions are updated to accept ProblemData)
        self._register_operators()

        # Generate Initial Solution (Greedy / Clarke-Wright)
        self.init_solution = vrp.initial.nearest_neighbor_tw(self.problem_data)
        
        self.current_solution = self.init_solution.copy()
        self.pre_solution = self.init_solution.copy()
        self.best_solution = self.init_solution.copy()
        
        # Normalize cost for feature extraction
        self.initial_cost = self.init_solution.objective()
        if self.initial_cost == 0: self.initial_cost = 1.0 # Avoid division by zero

        # Extract initial state features
        self.ppo_state = self._extract_features()
        
        return self.ppo_state.to_array(), {}

    def step(self, action):
        """Execute one step of the PPO agent."""
        d_idx, r_idx, accept, stop = action
        self.iters += 1
        
        # Track usage
        self.destroy_usage[d_idx] += 1
        self.repair_usage[r_idx] += 1
        
        # Execute ALNS Iteration
        # (Assuming alns.iterate is updated to handle new Solution structure)
        self.pre_solution, self.current_solution = self.alns.iterate(
            self.current_solution, self.pre_solution, d_idx, r_idx, accept, 
            data=self.problem_data # Pass problem data to operators
        )

        # Calculate Reward
        reward = self._calculate_reward(accept)
        
        # Check Best Solution
        if self.current_solution.objective() < self.best_solution.objective():
            self.best_solution = self.current_solution.copy()
            self.stagnation_counter = 0
            reward += 1.0
        else:
            self.stagnation_counter += 1

        # Update State Features
        self.ppo_state = self._extract_features()
        
        # Check Termination
        terminated = (stop == 1 and self.stop_counter >= 2) or (self.iters >= alns_config.num_iterations)
        truncated = False
        
        info = {}
        if terminated:
            info = {
                "best_cost": self.best_solution.objective(),
                "final_routes": self.best_solution.routes
            }

        return self.ppo_state.to_array(), reward, terminated, truncated, info

    def _extract_features(self) -> PPOState:
        """
        Calculates the 21+ features for the PPO neural network.
        Dimension-Agnostic: Works for any number of customers.
        """
        # 1. Solution Statistics
        routes = self.current_solution.routes # List of route objects/dicts
        num_routes = len(routes)
        
        total_cap_util = 0.0
        if num_routes > 0:
            for r in routes:
                # Assuming route structure: {'vehicle_type_id': 0, 'load_kg': 500, ...}
                v_type = self.problem_data.vehicle_types[r['vehicle_type_id']]
                if v_type.capacity_kg > 0:
                    total_cap_util += (r['load_kg'] / v_type.capacity_kg)
            avg_util = total_cap_util / num_routes
        else:
            avg_util = 0.0

        # 2. Instance Statistics (Calculated on the fly or cached)
        # Simple spatial density proxy: Mean distance from customers to depot
        avg_dist_to_depot = np.mean(self.problem_data.dist_matrix[0, 1:])
        density_proxy = self.problem_data.num_nodes / (avg_dist_to_depot + 1.0)

        # 3. Construct PPOState
        return PPOState(
            search_progress = self.iters / alns_config.num_iterations,
            stagnation_norm = min(1.0, self.stagnation_counter / self.stagnation_threshold),
            best_cost_norm = self.best_solution.objective() / self.initial_cost,
            current_cost_norm = self.current_solution.objective() / self.initial_cost,
            improvement_history = 0.0, # Implement moving average later
            
            demands_mean = np.mean(self.problem_data.demands_kg),
            demands_std = np.std(self.problem_data.demands_kg),
            tw_width_mean = np.mean(self.problem_data.time_windows[:, 1] - self.problem_data.time_windows[:, 0]),
            tw_tightness = 0.0, # Implement threshold logic later
            spatial_density = density_proxy,
            
            avg_capacity_utilization = avg_util,
            num_routes_norm = num_routes / self.problem_data.num_nodes, # Normalized against total nodes
            num_unassigned_norm = len(self.current_solution.unassigned) / self.problem_data.num_nodes,
            
            destroy_probs = self.destroy_usage / (np.sum(self.destroy_usage) + 1e-6),
            repair_probs = self.repair_usage / (np.sum(self.repair_usage) + 1e-6)
        )

    def _calculate_reward(self, accept_action):
        """Simplified reward function based on cost improvement."""
        # Calculate improvement percentage
        prev_cost = self.pre_solution.objective()
        curr_cost = self.current_solution.objective()
        delta = (prev_cost - curr_cost) / prev_cost if prev_cost > 0 else 0
        
        # Base reward on improvement
        reward = delta * 10.0
        
        # Penalty for rejecting a good move or accepting a bad one can be added here
        return reward

    def _register_operators(self):
        """Helper to register ALNS operators with the current problem data."""
        # Destroy operators
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data))
        self.alns.add_destroy_operator(vrp.create_worst_removal_operator(self.problem_data))
        # ... add others ...
        
        # Repair operators
        self.alns.add_repair_operator(vrp.create_greedy_repair_operator(self.problem_data))
        self.alns.add_repair_operator(vrp.create_regret_repair_operator(self.problem_data))
        # ... add others ...