# --- START OF FILE rvrpenv.py --- ver 5
# ver 5.1: thêm action maskign - Hùng

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

from core.data_structures import ProblemData, PPOState, RvrpState, Route
from core.real_data_loader import RealDataLoader
from config import PPOConfig, ALNSConfig
import alns as vrp
from alns.alns4ppo import ALNS4PPO
from sb3_contrib.common.maskable.utils import get_action_masks

ppo_config = PPOConfig()
alns_config = ALNSConfig()

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
        self.STOP_THRESHOLD = ppo_config.stop_threshold
        self.MAX_ITERATIONS = alns_config.num_iterations
        
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
        self.last_improvement = 0.0 # [UPDATED] Track improvement history
        
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
        
        self.alns.reset_opt()
        self._register_operators()

        self.init_solution = vrp.initial.clarke_wright_heterogeneous(self.problem_data)
        
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
        
        # [UPDATED] Update History for State
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

        return self.ppo_state.to_array(), reward, terminated, truncated, info

    def _extract_features(self) -> PPOState:
        sol = self.current_solution
        num_routes = len(sol.routes)
        
        # [UPDATED] Lấy giá trị thật từ ProblemData và RvrpState
        return PPOState(
            search_progress = self.iters / alns_config.num_iterations,
            stagnation_norm = min(1.0, self.stagnation_counter / 50.0),
            best_cost_norm = self.best_solution.objective() / self.initial_cost,
            current_cost_norm = sol.objective() / self.initial_cost,
            improvement_history = self.last_improvement,
            
            demands_mean = np.mean(self.problem_data.demands_kg),
            demands_std = np.std(self.problem_data.demands_kg),
            tw_width_mean = np.mean(self.problem_data.time_windows[:, 1] - self.problem_data.time_windows[:, 0]),
            
            # --- REAL FEATURES HERE ---
            tw_tightness = self.problem_data.static_tw_tightness,
            spatial_density = self.problem_data.static_spatial_density,
            max_wait_time_ratio = sol.max_wait_time_ratio, # Dynamic form solution
            # --------------------------
            
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
        reward_cost = improvement * ppo_config.reward_cost_scale
        
        # 2. Utilization Bonus (Secondary)
        prev_util = self.pre_solution.mean_capacity_utilization
        curr_util = self.current_solution.mean_capacity_utilization
        reward_util = (curr_util - prev_util) * ppo_config.reward_util_lambda
        
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
        # Destroy
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data, ratio=0.1))
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data, ratio=0.25))
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data, ratio=0.4))
        self.alns.add_destroy_operator(vrp.create_worst_removal_operator(self.problem_data, ratio=0.1))
        self.alns.add_destroy_operator(vrp.create_worst_removal_operator(self.problem_data, ratio=0.3))
        self.alns.add_destroy_operator(vrp.create_random_route_removal_operator(self.problem_data, ratio=0.1))
        
        # Repair
        self.alns.add_repair_operator(vrp.create_greedy_repair_operator(self.problem_data))
        self.alns.add_repair_operator(vrp.create_criticality_repair_operator(self.problem_data))
        self.alns.add_repair_operator(vrp.create_regret_repair_operator(self.problem_data))

    def _check_done(self, done_num):
        if done_num == 1: self.stop_counter += 1
        else: self.stop_counter = 0
        return self.stop_counter >= self.STOP_THRESHOLD or self.iters >= self.MAX_ITERATIONS

    def valid_action_mask(self) -> np.ndarray:
        """
        Tạo mask cho MaskablePPO. Trả về 1 array boolean phẳng (concatenated).
        True = Valid Action, False = Invalid Action.
        """
        # 1. Init masks
        # Destroy mask (Size = self.d_op_num)
        d_mask = np.ones(self.d_op_num, dtype=bool)
        
        # Repair mask (Size = self.r_op_num)
        r_mask = np.ones(self.r_op_num, dtype=bool)
        
        # Accept mask (Size = 2: Greedy, Explore) - Always Valid
        acc_mask = np.ones(2, dtype=bool)
        
        # Stop mask (Size = 2: Continue, Stop) - Always Valid
        stop_mask = np.ones(2, dtype=bool)

        # 2. Logic Masking
        current_routes_count = len(self.current_solution.routes)
        
        # Rule: Không thể dùng Route Removal (Index 5) nếu số route < 1
        # Index 5 được hardcode dựa trên thứ tự đăng ký trong _register_operators
        # (0: Rand1, 1: Rand2, 2: Rand3, 3: Worst1, 4: Worst2, 5: RouteRemoval)
        if current_routes_count < 1:
            if self.d_op_num > 5: # Safety check index
                d_mask[5] = False 
        
        # Rule: Nếu không có unassigned customer, Repair nào cũng chạy được (sẽ return no-op),
        # nhưng về lý thuyết không cần mask repair.
        
        # 3. Concatenate (Quan trọng cho MultiDiscrete của SB3)
        return np.concatenate([d_mask, r_mask, acc_mask, stop_mask])