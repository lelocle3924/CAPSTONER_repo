import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np

from config import PPOConfig

# --- CẤU HÌNH ---
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

ppo_config = PPOConfig()
# Các nhãn tên toán tử để vẽ biểu đồ cho đẹp
# Đảm bảo thứ tự khớp với cách bạn thêm toán tử trong vrptwenv.py
DESTROY_OPERATORS = [
    "Random Cust.", "Random Route", "String", "Worst", "Sequence"
]
REPAIR_OPERATORS = [
    "Greedy", "Criticality", "Regret"
]

# --- CÁC HÀM VẼ BIỂU ĐỒ ---

# Trong file plot_results.py

def plot_learning_curve(monitor_csv_path):
    """Vẽ biểu đồ phần thưởng trung bình trong quá trình huấn luyện."""
    if not os.path.exists(monitor_csv_path):
        print(f"Warning: Monitor file not found at '{monitor_csv_path}'. Skipping learning curve plot.")
        return
        
    try:
        df = pd.read_csv(monitor_csv_path, skiprows=1)
        if 'r' not in df.columns or 'l' not in df.columns:
            print(f"Warning: CSV file at '{monitor_csv_path}' does not contain 'r' or 'l' columns.")
            return

        # --- THAY ĐỔI Ở ĐÂY ---
        # 1. Tạo cột mới chứa tổng số bước tích lũy
        df['cumulative_steps'] = df['l'].cumsum()
        
        # 2. Làm mịn dữ liệu phần thưởng
        df['r_smooth'] = df['r'].rolling(window=50, min_periods=1).mean()

        plt.figure(figsize=(12, 6))
        # 3. Dùng cột mới để vẽ trục X
        plt.plot(df['cumulative_steps'], df['r_smooth'], label='Smoothed Reward')
        plt.title('Learning Curve (Average Reward over Time)')
        plt.xlabel('Cumulative Training Steps') # Cập nhật nhãn trục X
        plt.ylabel('Average Episode Reward')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(PLOTS_DIR, "1_learning_curve.png"))
        plt.close()
        print("Saved: 1_learning_curve.png")
    except Exception as e:
        print(f"Error plotting learning curve: {e}")


def plot_costs_comparison(ppo_costs_path, baseline_costs_path):
    """Vẽ biểu đồ hộp so sánh chi phí giữa PPO và Baseline."""
    if not (os.path.exists(ppo_costs_path) and os.path.exists(baseline_costs_path)):
        print("Warning: Cost files not found. Skipping cost comparison plots.")
        return
        
    df_ppo = pd.read_csv(ppo_costs_path)
    df_baseline = pd.read_csv(baseline_costs_path)

    df_ppo['Method'] = 'PPO-ALNS'
    df_baseline['Method'] = 'Baseline ALNS'

    df_combined = pd.concat([df_ppo, df_baseline], ignore_index=True)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Method', y='cost', data=df_combined)
    sns.stripplot(x='Method', y='cost', data=df_combined, color=".25", size=6)
    plt.title('Cost Comparison: PPO-ALNS vs. Baseline ALNS')
    plt.ylabel('Total Cost')
    plt.xlabel('Algorithm')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(PLOTS_DIR, "2_costs_boxplot.png"))
    plt.close()
    print("Saved: 2_costs_boxplot.png")
    
    print("\n--- Cost Statistics ---")
    ppo_mean, ppo_std = df_ppo['cost'].mean(), df_ppo['cost'].std()
    base_mean, base_std = df_baseline['cost'].mean(), df_baseline['cost'].std()
    print(f"PPO-ALNS Average Cost:      {ppo_mean:.4f} (+/- {ppo_std:.4f})")
    print(f"Baseline ALNS Average Cost: {base_mean:.4f} (+/- {base_std:.4f})")
    if base_mean > 0:
        improvement = (base_mean - ppo_mean) / base_mean * 100
        print(f"Average Improvement: {improvement:.2f}%")


def plot_operator_usage(behavior_log_path):
    """Vẽ biểu đồ tần suất sử dụng các toán tử."""
    if not os.path.exists(behavior_log_path):
        print("Warning: Behavior log not found. Skipping operator usage plots.")
        return
        
    df = pd.read_csv(behavior_log_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    destroy_counts = df['destroy_op'].value_counts().sort_index()
    destroy_labels = [DESTROY_OPERATORS[i] for i in destroy_counts.index]
    ax1.bar(destroy_labels, destroy_counts.values, color='skyblue')
    ax1.set_title('Destroy Operator Usage Frequency')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(destroy_labels, rotation=45, ha='right')
    
    repair_counts = df['repair_op'].value_counts().sort_index()
    repair_labels = [REPAIR_OPERATORS[i] for i in repair_counts.index]
    ax2.bar(repair_labels, repair_counts.values, color='salmon')
    ax2.set_title('Repair Operator Usage Frequency')
    ax2.set_xticklabels(repair_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "3_operator_usage.png"))
    plt.close()
    print("Saved: 3_operator_usage.png")


def plot_operator_heatmap(behavior_log_path):
    """Vẽ heatmap tương quan giữa các cặp toán tử."""
    if not os.path.exists(behavior_log_path):
        print("Warning: Behavior log not found. Skipping operator heatmap.")
        return

    df = pd.read_csv(behavior_log_path)
    
    contingency_table = pd.crosstab(df['destroy_op'], df['repair_op'])
    contingency_table.index = [DESTROY_OPERATORS[i] for i in contingency_table.index]
    contingency_table.columns = [REPAIR_OPERATORS[i] for i in contingency_table.columns]

    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='viridis', linewidths=.5)
    plt.title('Destroy-Repair Operator Co-occurrence Heatmap')
    plt.ylabel('Destroy Operator')
    plt.xlabel('Repair Operator')
    plt.savefig(os.path.join(PLOTS_DIR, "4_operator_heatmap.png"))
    plt.close()
    print("Saved: 4_operator_heatmap.png")


def plot_solution_routes(ax, title, node_coord, routes):
    """Hàm helper để vẽ các tuyến đường của một lời giải."""
    ax.set_title(title, fontsize=14)
    ax.scatter(node_coord[1:, 0], node_coord[1:, 1], c='black', s=20, label='Customers', zorder=3)
    ax.scatter(node_coord[0, 0], node_coord[0, 1], c='red', marker='s', s=100, label='Depot', zorder=5)

    colors = plt.get_cmap('gist_rainbow', len(routes))
    for i, route in enumerate(routes):
        if not route: continue
        full_route = [0] + route + [0]
        route_coords = node_coord[full_route]
        ax.plot(route_coords[:, 0], route_coords[:, 1], color=colors(i), marker='o', linestyle='-', markersize=4, label=f'Route {i+1}')
    
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small', loc='upper right')

def plot_route_comparison(instance_id=0):
    """Vẽ so sánh tuyến đường cho một instance cụ thể."""
    instance_path = os.path.join(RESULTS_DIR, f"instance_{instance_id}.pkl")
    ppo_sol_path = os.path.join(RESULTS_DIR, f"ppo_solution_{instance_id}.pkl")
    base_sol_path = os.path.join(RESULTS_DIR, f"baseline_solution_{instance_id}.pkl")

    if not all(os.path.exists(p) for p in [instance_path, ppo_sol_path, base_sol_path]):
        print(f"Warning: Files for instance ID {instance_id} not found. Skipping route plot.")
        return

    with open(instance_path, 'rb') as f: instance_data = pickle.load(f)
    with open(ppo_sol_path, 'rb') as f: ppo_solution = pickle.load(f)
    with open(base_sol_path, 'rb') as f: baseline_solution = pickle.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), sharex=True, sharey=True)
    fig.suptitle(f'Route Comparison for Instance {instance_id}', fontsize=16)

    plot_solution_routes(ax1, f'PPO-ALNS (Cost: {ppo_solution.objective():.2f})', instance_data['node_coord'], ppo_solution.routes)
    plot_solution_routes(ax2, f'Baseline ALNS (Cost: {baseline_solution.objective():.2f})', instance_data['node_coord'], baseline_solution.routes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(PLOTS_DIR, f"5_route_comparison_instance_{instance_id}.png"))
    plt.close()
    print(f"Saved: 5_route_comparison_instance_{instance_id}.png")


# --- HÀM MAIN ĐỂ CHẠY TẤT CẢ ---
if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("--- Starting Result Plotting ---")
    
    # Nhóm 1: Phân tích quá trình huấn luyện
    # THAY ĐỔI ĐƯỜNG DẪN NÀY cho phù hợp với thư mục log của bạn
    # Ví dụ: "logs/PPO_ALNS_1/monitor.csv"
    from config import PPOConfig
    ppo_config = PPOConfig()
    #MONITOR_LOG_PATH = os.path.join(ppo_config.tensorboard_log, "monitor.csv")
    MONITOR_LOG_PATH = ppo_config.monitor_path
    plot_learning_curve(MONITOR_LOG_PATH)
    
    # Nhóm 2 & 3: Phân tích kết quả và hành vi
    ppo_costs_file = os.path.join(RESULTS_DIR, "ppo_costs.csv")
    baseline_costs_file = os.path.join(RESULTS_DIR, "baseline_costs.csv")
    behavior_log_file = os.path.join(RESULTS_DIR, "ppo_behavior_log.csv")
    
    
    plot_costs_comparison(ppo_costs_file, baseline_costs_file)
    plot_operator_usage(behavior_log_file)
    plot_operator_heatmap(behavior_log_file)

    # Nhóm 4: Phân tích tuyến đường
    plot_route_comparison(instance_id=0) # Vẽ cho instance đầu tiên
    
    print("\n--- Plotting Complete ---")