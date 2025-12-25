import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import modules từ project
from ppo.rvrpenv import RVRPEnvironment
from core.data_structures import RvrpState, Route, ProblemData

# --- CONFIG ---
ORDER_PATH = "inputs/CleanData/Split_TransportOrder_allabove1_2524.csv"
TRUCK_PATH = "inputs/MasterData/TruckMaster.csv"
timestamp = datetime.now().strftime("%d%m_%H%M")
ENABLE_REALTIME_PLOT = True 
OUTPUT_IMG_DIR = f"output/real_time_{timestamp}"
LOG_FILE = f"manual_test_log_{timestamp}.txt"

# --- VISUALIZER CLASS (Lightweight for Real-time) ---
class GameVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.ion() # Bật chế độ Interactive (không block code)
        self.colors = {
            'MC': 'tab:green',
            'AUV': 'gold',
            '4w': 'tab:blue',
            '6w': 'tab:purple',
            '10w': 'tab:red',
            '40ft': 'black'
        }
        self.fallback_colors = ['cyan', 'magenta', 'lime']
        
        # Tạo folder output
        if not os.path.exists(OUTPUT_IMG_DIR):
            os.makedirs(OUTPUT_IMG_DIR)

    def _get_color(self, v_name):
        for key, c in self.colors.items():
            if key in v_name: return c
        return 'gray'

    def update_plot(self, state: RvrpState, data: ProblemData, step_info: str, step_idx: int):
        self.ax.clear()
        
        # [FIX]: Longitude = X, Latitude = Y
        
        # 1. Vẽ Depot
        depot = data.coords[0] # [Lat, Long]
        self.ax.scatter(depot[1], depot[0], c='red', marker='s', s=100, zorder=10, label='Depot')
        
        # 2. Vẽ Unassigned Customers
        if state.unassigned:
            un_coords = data.coords[state.unassigned]
            self.ax.scatter(un_coords[:, 1], un_coords[:, 0], c='black', marker='x', s=50, label='Unassigned')

        # 3. Vẽ Routes
        legend_patches = []
        seen_types = set()
        
        for route in state.routes:
            v_name = route.vehicle_type.name
            color = self._get_color(v_name)
            
            # Lấy tọa độ: Depot -> Nodes -> Depot
            path_idxs = [0] + route.node_sequence + [0]
            path_coords = data.coords[path_idxs]
            
            # Vẽ đường (Long = col 1, Lat = col 0)
            self.ax.plot(path_coords[:, 1], path_coords[:, 0], c=color, linewidth=1.5, alpha=0.7)
            # Vẽ điểm khách hàng
            self.ax.scatter(path_coords[1:-1, 1], path_coords[1:-1, 0], c=color, s=20)
            
            if v_name not in seen_types:
                seen_types.add(v_name)
                legend_patches.append(mpatches.Patch(color=color, label=v_name))

        # 4. Trang trí
        self.ax.set_title(f"STEP {step_idx}: {step_info}\nCost: {state.objective():,.0f} | Routes: {len(state.routes)} | Unassigned: {len(state.unassigned)}")
        self.ax.legend(handles=legend_patches, loc='upper right')
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
        # 5. Render & Save
        plt.draw()
        plt.pause(0.1) # Dừng 0.1s để render
        
        # Save frame
        save_path = os.path.join(OUTPUT_IMG_DIR, f"real_time_{step_idx}.png")
        plt.savefig(save_path)

    def close(self):
        plt.ioff()
        plt.show() # Giữ cửa sổ lại lúc kết thúc

# --- HELPER FUNCTIONS ---
def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")

def log_to_file(file_handle, text):
    file_handle.write(text + "\n")
    file_handle.flush()

def print_operators(env):
    print("\n--- DESTROY ---")
    for i, (name, _) in enumerate(env.alns.destroy_operators):
        print(f"  [{i}] {name}")
    print("\n--- REPAIR ---")
    for i, (name, _) in enumerate(env.alns.repair_operators):
        print(f"  [{i}] {name}")
    print("-" * 40)

def print_solution_state(tag: str, solution: RvrpState, env: RVRPEnvironment, file_handle):
    summary = []
    summary.append(f"[{tag}] SUMMARY:")
    summary.append(f"  Total Cost: {solution.objective():,.2f}")
    summary.append(f"  Total Routes: {len(solution.routes)}")
    summary.append(f"  Unassigned: {len(solution.unassigned)}")
    summary.append(f"  Mean Util: {solution.mean_capacity_utilization*100:.2f}%")
    
    summary.append("  > ROUTE DETAILS:")
    for i, route in enumerate(solution.routes):
        v = route.vehicle_type
        load_str = f"{route.total_load_kg:.0f}/{v.capacity_kg:.0f}kg"
        stops = len(route.node_sequence)
        marker = ">>>" if v.capacity_kg >= 2000 else "   " 
        summary.append(f"    {marker} R{i+1:02d}: Type={v.name:<5} | Load={load_str:<12} | Stops={stops:<3} | Cost={route.cost:,.0f}")

    output = "\n".join(summary)
    print(output)
    log_to_file(file_handle, output)

# --- MAIN ---
def main():
    print_header("INITIALIZING ENVIRONMENT")
    if not os.path.exists(ORDER_PATH) or not os.path.exists(TRUCK_PATH):
        print("❌ Error: Input files not found.")
        return

    f_log = open(LOG_FILE, "w", encoding="utf-8")
    log_to_file(f_log, f"TEST SESSION START: {datetime.now()}\n")
    log_to_file(f_log, f"Order File: {ORDER_PATH}")
    # Init Env
    env = RVRPEnvironment(ORDER_PATH, TRUCK_PATH, is_test_mode=False)
    
    # Init Visualizer
    viz = None
    if ENABLE_REALTIME_PLOT:
        print("🖥️  Real-time Visualization: ON")
        viz = GameVisualizer()
    
    # 1. Generate Initial Solution
    print("Generating Initial Solution...")
    obs, _ = env.reset()
    
    print_header("INITIAL SOLUTION")
    print_solution_state("INIT", env.current_solution, env, f_log)
    
    if viz:
        viz.update_plot(env.current_solution, env.problem_data, "Initial Solution", 0)

    # 2. Main Loop
    step_count = 0
    total_reward = 0.0
    
    while True:
        print_header(f"ITERATION {step_count + 1}")
        print_operators(env)
        
        print("\nINPUT ACTION (Format: d_idx r_idx accept[0/1] stop[0/1])")
        print("Example: 0 0 1 0")
        user_input = input(">> Your Action ('q' to quit): ").strip()
        
        if user_input.lower() == 'q':
            break
            
        try:
            parts = list(map(int, user_input.split()))
            if len(parts) != 4:
                print("❌ Invalid format.")
                continue
            
            d_idx, r_idx, acc, stop = parts
            
            # Validate
            # if d_idx < 0 or d_idx >= env.d_op_num:
            #     print(f"❌ Destroy index must be 0-{env.d_op_num-1}")
            #     continue
            # if r_idx < 0 or r_idx >= env.r_op_num:
            #     print(f"❌ Repair index must be 0-{env.r_op_num-1}")
            #     continue

            action = np.array([d_idx, r_idx, acc, stop], dtype=np.int32)
            
            # Log Name
            d_name = env.alns.destroy_operators[d_idx][0]
            r_name = env.alns.repair_operators[r_idx][0]
            acc_str = "EXPLORE" if acc == 1 else "GREEDY"
            
            log_str = f"\n>>> EXECUTE: {d_name} -> {r_name} ({acc_str})"
            print(log_str)
            log_to_file(f_log, log_str)
            
            # --- STEP ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            print(f"  -> Reward: {reward:.4f}")
            print_solution_state(f"STEP {step_count}", env.current_solution, env, f_log)
            
            # --- UPDATE PLOT ---
            if viz:
                step_info = f"{d_name}\n-> {r_name} ({acc_str})"
                viz.update_plot(env.current_solution, env.problem_data, step_info, step_count)
            
            if terminated or truncated or stop == 1:
                print("\n🛑 STOP SIGNAL RECEIVED.")
                break
                
        except ValueError:
            print("❌ Invalid input (numbers only).")
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            break

    print_header("SESSION END")
    print(f"Steps: {step_count} | Total Reward: {total_reward:.4f}")
    print(f"Images saved to: {OUTPUT_IMG_DIR}")
    f_log.close()
    
    if viz:
        print("Close the plot window to exit...")
        viz.close()

if __name__ == "__main__":
    main()