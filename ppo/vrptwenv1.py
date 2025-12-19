import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import PPOConfig, ALNSConfig

from ppo.instance_pool import InstancePool
import alns.vrp as vrp
from alns.alns4ppo import ALNS4PPO

ppo_config = PPOConfig()
alns_config = ALNSConfig()

MAX_ITERATIONS = alns_config.num_iterations
STOP_THRESHOLD = ppo_config.stop_threshold

class VRPTWEnvironment(gym.Env):
    def __init__(
        self, instance_pool: InstancePool, record_data: bool = False, alns_init: int = 0, is_test_mode: bool = False
    ):
        super(VRPTWEnvironment, self).__init__()

        self.instance_pool = instance_pool
        self.alns = ALNS4PPO()
        self.dimension = ppo_config.dimension
        self.d_op_num = alns_config.num_destroy
        self.r_op_num = alns_config.num_repair
        self.destroy_usage = np.zeros(self.d_op_num, dtype=np.float32)
        self.repair_usage = np.zeros(self.r_op_num, dtype=np.float32)
        self.iters = 0
        self.stop_counter = 0
        self.record_data = record_data
        self.alns_init = alns_init
        
        self.is_test_mode = is_test_mode
        self.behavior_log = []
        self.instance_data = {}

        self.action_space = spaces.MultiDiscrete([self.d_op_num, self.r_op_num, 2, 2])

        self.observation_space = spaces.Dict({
            "search_progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "solution_delta": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "init_cost": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "best_cost": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "destroy_usage": spaces.Box(low=0, high=1, shape=(self.d_op_num,), dtype=np.float32),
            "repair_usage": spaces.Box(low=0, high=1, shape=(self.r_op_num,), dtype=np.float32),
            "demand": spaces.Box(low=0, high=1, shape=(self.dimension,), dtype=np.float32),
            "time_windows": spaces.Box(low=0, high=1, shape=(self.dimension, 2), dtype=np.float32),
            "service_times": spaces.Box(low=0, high=1, shape=(self.dimension,), dtype=np.float32),
            "travel_times": spaces.Box(low=0, high=1, shape=(self.dimension, self.dimension), dtype=np.float32),
        })
        self.state = {} # Sẽ được khởi tạo trong reset

    def reset(self, seed=None, options=None):
        
        #============================================================
        data = self.instance_pool.sample()
        #============================================================
        
        self.instance_data = data
        if self.record_data: self.data = data
        
        self.alns.reset_opt()
        # kể tên cả opts
        d_opts_all = [
            vrp.create_random_customer_removal_operator(data), vrp.create_random_route_removal_operator(),
            vrp.create_string_removal_operator(data), vrp.create_worst_removal_operator(data),
            vrp.create_sequence_removal_operator(data), vrp.create_related_removal_operator(data)
        ]
        r_opts_all = [
            vrp.create_greedy_repair_operator(data), vrp.create_criticality_repair_operator(data),
            vrp.create_regret_repair_operator(data)
        ]
        for op in d_opts_all[:self.d_op_num]: self.alns.add_destroy_operator(op) # add opts vào instance heuristics
        for op in r_opts_all[:self.r_op_num]: self.alns.add_repair_operator(op)

        if self.alns_init == 0: self.init_solution = vrp.clarke_wright_tw(data=data)
        else: self.init_solution = vrp.initial.nearest_neighbor_tw(data=data)  # tính initial solution

        self.max_cost = np.sum(data["travel_times"][0, 1:]) * 2
        init_cost_norm = self.init_solution.objective() / self.max_cost if self.max_cost > 0 else 0 # normalize init cost
        
        self.state = {
            "search_progress": np.array([0.0], dtype=np.float32),
            "solution_delta": np.array([0.0], dtype=np.float32),
            "init_cost": np.array([init_cost_norm], dtype=np.float32),
            "best_cost": np.array([init_cost_norm], dtype=np.float32),
            "destroy_usage": np.zeros(self.d_op_num, dtype=np.float32),
            "repair_usage": np.zeros(self.r_op_num, dtype=np.float32),
            "demand": np.array(data["demand"], dtype=np.float32),
            "time_windows": np.array(data["time_windows"], dtype=np.float32),
            "service_times": np.array(data["service_times"], dtype=np.float32),
            "travel_times": np.array(data["travel_times"], dtype=np.float32),
        } # khởi tạo state object

        self.destroy_usage.fill(0)
        self.repair_usage.fill(0)
        self.current_solution = self.init_solution.copy()
        self.pre_solution = self.init_solution.copy()
        self.best_solution = self.init_solution.copy()
        self.iters = 0
        self.stop_counter = 0
        self.alns._rng = np.random.default_rng(seed)

        if self.is_test_mode:
            self.behavior_log.clear()

        return self.state, {}

    def step(self, action):
        d_idx, r_idx, accept, stop = action
        
        # ghi lại logger nếu mode test
        if self.is_test_mode:
            log_entry = {
                'timestep': self.iters + 1,
                'search_progress': self.state.get('search_progress', [0.0])[0],
                'destroy_op': d_idx,
                'repair_op': r_idx,
                'accepted': accept,
                'stopped': stop,
                'reward': 0, # Sẽ được cập nhật sau
                'solution_delta': self.state.get('solution_delta', [0.0])[0]
            }
            self.behavior_log.append(log_entry)
        
        self.iters += 1
        
        #========================================== WHY
        reward = 0  # khởi tạo reward
        #========================================== WHY --> COULD ONLY POSSIBLY GET REWARD IF THE ROUTE TERMINATES. AKA, REWARD IS ONLY GIVEN WHEN THE AGENT REACHES TERMINAL STATE, NO INTERMEDIATE REWARD

        self.destroy_usage[d_idx] += 1 # dùng opts nào tăng số lần dùng opt đó lên
        self.repair_usage[r_idx] += 1
        
        improvement = self.state.get('solution_delta', [0.0])[0] # lục trong state dict và lấy ra sol_delta
        # punishment for not accepting better solution
        if improvement > 0 and accept == 0: reward -= 0.8 * improvement
        # punishment for accepting worse solution, since improvement is negative
        if improvement < 0 and accept == 1: reward += 0.8 * improvement

        # ĐÂY LÀ CHỖ DUY NHẤT CÓ ĐỤNG ĐẾN CvrptwState
        self.pre_solution, self.current_solution = self.alns.iterate(self.current_solution, self.pre_solution, d_idx, r_idx, accept)

        best_cost = self.best_solution.objective()
        pre_cost = self.pre_solution.objective()
        cur_cost = self.current_solution.objective()
        
        improvement = (pre_cost - cur_cost) / pre_cost if pre_cost != 0 else 0
        # FOUND NEW BEST SOLUTION
        if cur_cost < best_cost:
            self.best_solution = self.current_solution.copy()
            relative_to_best = (best_cost - cur_cost) / best_cost if best_cost != 0 else 0
            reward += 2 * relative_to_best
        
        # Cập nhật lại reward trong log
        if self.is_test_mode:
            self.behavior_log[-1]['reward'] = reward

        # update thông số trong ppo_state, không phải CvrptwState
        self._update_state(improvement)

        terminated = self._check_done(stop)
        truncated = False
        
        info = {}
        #Calculate final reward if terminated
        if terminated:
            init_cost_norm = self.state.get('init_cost', [0.0])[0]
            best_cost_norm = self.state.get('best_cost', [0.0])[0]
            final_reward = 4 * (init_cost_norm - best_cost_norm) # FINAL REWARD SAU KHI ĐÃ CHẠY 1 ĐỐNG ALNS = INIT_COST_NORM - BEST_COST_NORM
            reward += final_reward
            reward -= 0.04 * self.state.get('search_progress', [0.0])[0]
            info = {
                "best_solution": self.best_solution, 
                "best_cost": self.best_solution.objective(),
                "instance_data": self.instance_data, 
                "behavior_log": self.behavior_log.copy()
            }

        return self.state, reward, terminated, truncated, info

    def _update_state(self, improvement):
        if not self.state: return
        self.state["search_progress"][0] = self.iters / MAX_ITERATIONS
        self.state["solution_delta"][0] = improvement
        self.state["best_cost"][0] = self.best_solution.objective() / self.max_cost if self.max_cost > 0 else 0
        total_destroy = np.sum(self.destroy_usage)
        total_repair = np.sum(self.repair_usage)
        if total_destroy > 0: self.state["destroy_usage"] = self.destroy_usage / total_destroy
        if total_repair > 0: self.state["repair_usage"] = self.repair_usage / total_repair

    def _check_done(self, done_num):
        if done_num == 1: self.stop_counter += 1
        else: self.stop_counter = 0
        return self.stop_counter >= STOP_THRESHOLD or self.iters >= MAX_ITERATIONS
        #check_done được gọi khi