# --- START OF FILE rvrpenv.py ---
# ver 6: optimization for gpu

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import time

from core.data_structures import ProblemData, PPOState, RvrpState, Route
from core.real_data_loader import RealDataLoader
from config import PPOConfig, ALNSConfig, PathConfig
import alns as vrp
from alns.alns4ppo import ALNS4PPO
from sb3_contrib.common.maskable.utils import get_action_masks

ppo_cfg = PPOConfig()
alns_cfg = ALNSConfig()

class RVRPEnvironment(gym.Env):
    def __init__(self, 
                 order_csv_path: str, 
                 truck_csv_path: str, 
                 is_test_mode: bool = False):
        super().__init__()
        
        # 1. Load Data
        self.loader = RealDataLoader()
        self.problem_data: ProblemData = self.loader.load_day_data(
            order_csv_path=order_csv_path,
            truck_csv_path=truck_csv_path
        )
        self.STOP_THRESHOLD = ppo_cfg.stop_threshold
        self.MAX_ITERATIONS = alns_cfg.num_iterations
        
        self.is_test_mode = is_test_mode
        self.alns = ALNS4PPO()
        
        # 2. Spaces
        self.alns.reset_opt()
        self._register_operators()
        self.d_op_num = len(self.alns.destroy_operators)
        self.r_op_num = len(self.alns.repair_operators)
        
        self.action_space = spaces.MultiDiscrete([self.d_op_num, self.r_op_num, 2, 2])
        
        obs_size = PPOState.get_observation_size(self.d_op_num, self.r_op_num)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # 3. Tracking
        self.iters = 0
        self.stop_counter = 0
        self.stagnation_counter = 0
        self.last_improvement = 0.0
        
        self.destroy_usage = np.zeros(self.d_op_num, dtype=np.float32)
        self.repair_usage = np.zeros(self.r_op_num, dtype=np.float32)
        
        # State Objects
        self.current_solution: RvrpState = None
        self.best_solution: RvrpState = None
        self.ppo_state: PPOState = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.iters = 0
        self.stop_counter = 0
        self.stagnation_counter = 0
        self.last_improvement = 0.0
        self.destroy_usage.fill(0)
        self.repair_usage.fill(0)

        self.static_features = np.array([
            np.mean(self.problem_data.demands_kg),
            np.std(self.problem_data.demands_kg),
            np.mean(self.problem_data.time_windows[:, 1] - self.problem_data.time_windows[:, 0]),
            self.problem_data.static_tw_tightness,
            self.problem_data.static_spatial_density
        ], dtype=np.float32)
        
        self.init_solution = vrp.initial.clarke_wright_heterogeneous(self.problem_data)
        
        self.current_solution = self.init_solution.copy()
        self.pre_solution = self.init_solution.copy()
        self.best_solution = self.init_solution.copy()
        
        self.initial_cost = self.init_solution.objective()
        if self.initial_cost == 0: self.initial_cost = 1.0

        self.ppo_state = self._extract_features()
        return self.ppo_state.to_array(), {}

    def step(self, action):
        start_time = time.time()
        d_idx, r_idx, accept, stop = action
        self.iters += 1
        
        self.destroy_usage[d_idx] += 1
        self.repair_usage[r_idx] += 1
        
        # 1. ALNS Iterate (Core Logic)
        # accept=0: Greedy (Reject if worse), accept=1: Explore (Accept even if worse)
        self.pre_solution, self.current_solution = self.alns.iterate(
            self.current_solution, self.pre_solution, d_idx, r_idx, accept, 
            data=self.problem_data
        )

        # 2. Calculate Metrics & Reward
        prev_cost = self.pre_solution.objective()
        curr_cost = self.current_solution.objective()
        best_cost = self.best_solution.objective()
        
        # Improvement relative to previous step (Pos: Better, Neg: Worse)
        base = prev_cost if prev_cost > 1e-6 else 1.0
        step_improvement = (prev_cost - curr_cost) / base
        
        self.last_improvement = step_improvement

        # 3. Update Global Best & Stagnation
        if curr_cost < best_cost:
            self.best_solution = self.current_solution.copy()
            self.stagnation_counter = 0
            # Bonus reward for finding new global best
            reward_best = 2.0 * (best_cost - curr_cost) / best_cost
        else:
            self.stagnation_counter += 1
            reward_best = 0.0

        # 4. Calculate Reward
        reward = self._calculate_reward(step_improvement, accept, self.stagnation_counter)
        reward += reward_best

        # 5. Prepare Next State
        self.ppo_state = self._extract_features()
        
        terminated = self._check_done(stop)
        truncated = False
        
        info = {}
        if terminated:
            info = {
                "best_cost": self.best_solution.objective(),
                "num_vehicles": len(self.best_solution.routes),
                "mean_utilization": self.ppo_state.mean_cap_utilization
            }

        end_time = time.time() # Kết thúc đo giờ
        step_duration = end_time - start_time
        if step_duration > 0.5 or self.iters % 10 == 0:
            print(f"[ENV_DEBUG] Step {self.iters}: Duration={step_duration:.4f}s | Op: D[{d_idx}]-R[{r_idx}]")

        return self.ppo_state.to_array(), reward, terminated, truncated, info

    def _extract_features(self) -> PPOState:
        sol = self.current_solution
        num_routes = len(sol.routes)
        
        dem_mean, dem_std, tw_mean, tw_tight, spat_dens = self.static_features

        return PPOState(
            search_progress = self.iters / alns_cfg.num_iterations,
            stagnation_norm = min(1.0, self.stagnation_counter / 50.0),
            best_cost_norm = self.best_solution.objective() / self.initial_cost,
            current_cost_norm = sol.objective() / self.initial_cost,
            improvement_history = self.last_improvement,
            
            demands_mean = dem_mean,
            demands_std = dem_std,
            tw_width_mean = tw_mean,
            tw_tightness = tw_tight,
            spatial_density = spat_dens,
            
            max_wait_time_ratio = sol.max_wait_time_ratio,
            min_cap_utilization = sol.min_capacity_utilization,
            mean_cap_utilization = sol.mean_capacity_utilization,
            max_cap_utilization = sol.max_capacity_utilization,
            
            num_routes_norm = num_routes / self.problem_data.num_nodes, 
            num_unassigned_norm = len(sol.unassigned) / self.problem_data.num_nodes,
            
            destroy_probs = self.destroy_usage / (np.sum(self.destroy_usage) + 1e-6),
            repair_probs = self.repair_usage / (np.sum(self.repair_usage) + 1e-6)
        )

    def _calculate_reward(self, improvement, accept_action, stagnation):
        """
        Custom Reward Function
        """
        # 1. Cost Improvement (Main Driver)
        # improvement > 0: Reward (Cost giảm)
        # improvement < 0: Penalty (Cost tăng)
        reward_cost = improvement * ppo_cfg.reward_cost_scale
        
        # 2. Utilization Bonus (Secondary)
        prev_util = self.pre_solution.mean_capacity_utilization
        curr_util = self.current_solution.mean_capacity_utilization
        reward_util = (curr_util - prev_util) * ppo_cfg.reward_util_lambda
        
        # 3. [UPDATED] Smart Exploration Bonus
        # Nếu đang bế tắc (stagnation cao) mà Agent dám Accept (1) dù kết quả tệ (improvement < 0)
        # -> Thưởng nhẹ để khuyến khích thoát cực trị địa phương.
        # Nếu không bế tắc mà Accept bừa bãi -> Không thưởng (vẫn bị phạt bởi reward_cost)
        reward_explore = 0.0
        if improvement < 0 and accept_action == 1:
            if stagnation > 10:
                reward_explore = 0.05 * (stagnation / 50.0) # Bonus tăng theo độ bế tắc
        
        return reward_cost + reward_util + reward_explore

    def _register_operators(self):
        r_small = 0.1
        r_medium = 0.3
        r_large = 0.5
        
        # --- 1. RANDOM REMOVAL (Basic Diversification) ---
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data, ratio=r_small), name="random_small")
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data, ratio=r_medium), name="random_medium")
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data, ratio=r_large), name="random_large")

        # --- 2. WORST REMOVAL (Cost Reduction Focus) ---
        self.alns.add_destroy_operator(vrp.create_worst_removal_operator(self.problem_data, ratio=r_small), name="worst_small")
        self.alns.add_destroy_operator(vrp.create_worst_removal_operator(self.problem_data, ratio=r_medium), name="worst_medium")
        self.alns.add_destroy_operator(vrp.create_worst_removal_operator(self.problem_data, ratio=r_large), name="worst_large")

        # --- 3. STRING REMOVAL (Spatial/Sequence Focus) ---
        self.alns.add_destroy_operator(vrp.create_string_removal_operator(self.problem_data, ratio=r_small), name="string_small")
        self.alns.add_destroy_operator(vrp.create_string_removal_operator(self.problem_data, ratio=r_medium), name="string_medium")
        self.alns.add_destroy_operator(vrp.create_string_removal_operator(self.problem_data, ratio=r_large), name="string_large")

        # --- 4. RELATED REMOVAL (Shaw - Similarity Focus) ---
        self.alns.add_destroy_operator(vrp.create_related_removal_operator(self.problem_data, ratio=r_small), name="related_small")
        self.alns.add_destroy_operator(vrp.create_related_removal_operator(self.problem_data, ratio=r_medium), name="related_medium")
        self.alns.add_destroy_operator(vrp.create_related_removal_operator(self.problem_data, ratio=r_large), name="related_large")

        # --- 5. ROUTE REMOVAL (Structure Focus - High Impact) ---
        self.alns.add_destroy_operator(vrp.create_random_route_removal_operator(self.problem_data, ratio=r_small), name="route_small")
        self.alns.add_destroy_operator(vrp.create_random_route_removal_operator(self.problem_data, ratio=r_medium), name="route_medium")
        self.alns.add_destroy_operator(vrp.create_random_route_removal_operator(self.problem_data, ratio=r_large), name="route_large")

        # --- 6. SEQUENCE REMOVAL (Sub-tour Focus) ---
        self.alns.add_destroy_operator(vrp.create_sequence_removal_operator(self.problem_data, ratio=r_small), name="sequence_small")
        self.alns.add_destroy_operator(vrp.create_sequence_removal_operator(self.problem_data, ratio=r_medium), name="sequence_medium")
        self.alns.add_destroy_operator(vrp.create_sequence_removal_operator(self.problem_data, ratio=r_large), name="sequence_large")

        # --- 7. LOW UTILIZATION ROUTE REMOVAL (Optimization Focus) ---
        self.alns.add_destroy_operator(vrp.create_low_utilization_route_removal_operator(self.problem_data, ratio=r_small), name="low_util_small")
        self.alns.add_destroy_operator(vrp.create_low_utilization_route_removal_operator(self.problem_data, ratio=r_medium), name="low_util_medium")
        self.alns.add_destroy_operator(vrp.create_low_utilization_route_removal_operator(self.problem_data, ratio=r_large), name="low_util_large")

        # --- 8. ELIMINATE SMALL ROUTES (Cleaner) ---
        self.alns.add_destroy_operator(vrp.create_eliminate_small_route_operator(self.problem_data, min_stops=3), name="eliminate_small")
        
        # --- REPAIR OPERATORS ---
        # 1. Greedy (Baseline)
        self.alns.add_repair_operator(vrp.create_greedy_repair_operator(self.problem_data), name="greedy_repair")
        
        # 2. Criticality (Composite Score)
        self.alns.add_repair_operator(vrp.create_criticality_repair_operator(self.problem_data), name="criticality_repair")
        
        # 3. Regret-2 (Lookahead 2)
        self.alns.add_repair_operator(vrp.create_regret_repair_operator(self.problem_data, k=2), name="regret_2_repair")
        
        # 4. Regret-3 (Lookahead 3)
        self.alns.add_repair_operator(vrp.create_regret_repair_operator(self.problem_data, k=3), name="regret_3_repair")
        
        # 5. GRASP (Randomized)
        self.alns.add_repair_operator(vrp.create_grasp_repair_operator(self.problem_data, rcl_size=3), name="grasp_3_repair")
        
        # 6. Farthest (Spatial - Outside In)
        self.alns.add_repair_operator(vrp.create_farthest_insertion_repair_operator(self.problem_data), name="farthest_repair")
        
        # 7. [NEW] Largest Demand (Bin Packing Focus)
        self.alns.add_repair_operator(vrp.create_largest_demand_repair_operator(self.problem_data), name="largest_demand_repair")
        
        # 8. [NEW] Earliest TW (Scheduling Focus)
        self.alns.add_repair_operator(vrp.create_earliest_tw_repair_operator(self.problem_data), name="earliest_tw_repair")
        
        # 9. [NEW] Closest (Spatial - Depot Focus)
        self.alns.add_repair_operator(vrp.create_closest_to_depot_repair_operator(self.problem_data), name="closest_depot_repair")


    def _check_done(self, done_num):
        if done_num == 1: self.stop_counter += 1
        else: self.stop_counter = 0
        return self.stop_counter >= self.STOP_THRESHOLD or self.iters >= self.MAX_ITERATIONS

    def valid_action_mask(self) -> np.ndarray:
        # 1. Init masks (Default True = Valid)
        d_mask = np.ones(self.d_op_num, dtype=bool)
        r_mask = np.ones(self.r_op_num, dtype=bool)
        acc_mask = np.ones(2, dtype=bool)
        stop_mask = np.ones(2, dtype=bool)

        # 2. Get State Info
        current_routes_count = len(self.current_solution.routes)
        
        # 3. Dynamic Masking Logic
        
        # --- DESTROY MASKING ---
        if current_routes_count == 0:
            # Nếu chưa có route nào, CẤM các hành động yêu cầu phải có route để xóa.
            # Duyệt qua danh sách operator thực tế để check tên
            for i, (op_name, _) in enumerate(self.alns.destroy_operators):
                if any(keyword in op_name for keyword in ["route", "sequence", "low_util", "eliminate"]):
                    d_mask[i] = False
        
        # 4. Concatenate and Return
        return np.concatenate([d_mask, r_mask, acc_mask, stop_mask])