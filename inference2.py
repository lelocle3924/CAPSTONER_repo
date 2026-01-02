import pandas as pd
import numpy as np
import os
import shutil
import glob
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from core.data_structures import RvrpState, ProblemData
from core.real_data_loader import RealDataLoader
from core.visualizer import RouteVisualizer
from config import PathConfig, PPOConfig
from ppo.rvrpenv import RVRPEnvironment
from core.distance_matrix import DistanceMatrixCalculator

from core.export_solution import export_solution_to_csv_and_map

path_cfg = PathConfig()
ppo_cfg = PPOConfig()

# --- 1. DATA PRE-PROCESSING ---

def split_orders_by_date(original_csv_path, output_dir):
    """
    Đọc file Order tổng -> Group đơn trùng -> Tách ra file theo ngày.
    """
    print(f">>> [1/4] Splitting Orders from {original_csv_path}...")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # Clear old temp files
    os.makedirs(output_dir)
    
    df = pd.read_csv(original_csv_path)
    
    # Normalize Date Column (Adjust 'Delivery Date' to match your CSV header)
    date_col = 'Delivery Date' 
    if date_col not in df.columns:
        # Fallback logic if column name differs
        possible_cols = [c for c in df.columns if 'date' in c.lower()]
        if possible_cols: date_col = possible_cols[0]
    
    unique_dates = df[date_col].unique()
    generated_files = []
    
    print(f"    Found {len(unique_dates)} unique dates.")
    
    for d in unique_dates:
        # Filter Day
        day_df = df[df[date_col] == d].copy()
        
        # Consolidate Logic: Group by Customer to sum Demand
        # Giữ lại thông tin tĩnh (Lat, Long, TimeWindow...) của dòng đầu tiên
        agg_rules = {
            'KGM': 'sum', 'CBM': 'sum',
            'CusLat': 'first', 'CusLong': 'first',
            'Beginning1': 'first', 'Ending1': 'first',
            'DwellTime': 'first', 'AllowedTrucks': 'first',
            'Depot': 'first', 'DepotLat': 'first', 'DepotLong': 'first'
        }
        # Add other columns to 'first' if needed to prevent loss
        for col in day_df.columns:
            if col not in agg_rules and col not in ['Customer', date_col]:
                agg_rules[col] = 'first'
                
        consolidated_df = day_df.groupby('Customer', as_index=False).agg(agg_rules)
        
        # Save Temp File
        safe_date = str(d).replace("/", "-").replace(" ", "_")
        fname = os.path.join(output_dir, f"orders_{safe_date}.csv")
        consolidated_df.to_csv(fname, index=False)
        generated_files.append((safe_date, fname))
        
    print(f"    Generated {len(generated_files)} temp files in {output_dir}")
    return generated_files

# --- 2. MATRIX WARM-UP ---

def warmup_and_get_master_id(full_order_path):
    """
    1. Chạy Loader trên file tổng để đảm bảo cache master tồn tại.
    2. Trả về Depot ID chuẩn (VD: '2513') để các bước sau dùng lại.
    """
    print(">>> [2/4] Warming up Master Distance Matrix...")
    
    # Init Loader tạm để lấy ID và trigger cache
    loader = RealDataLoader()
    # Load file tổng, không override (để nó tự tạo cache theo tên file gốc hoặc depot gốc)
    problem_data = loader.load_day_data(full_order_path, path_cfg.TRUCK_PATH)
    
    # Lấy Depot ID từ ID của node đầu tiên trong file tổng
    # Ví dụ: node_ids[0] là '2513'
    master_id = problem_data.node_ids[0]
    
    print(f"    Master Depot ID detected: {master_id}")
    print(f"    Cache file should be: distance_matrix_meters{master_id}.csv")
    return master_id

# --- 3. HELPER: ADAPTER FOR EXPORT ---

class ResultAdapter:
    """
    Chuyển đổi dữ liệu từ RvrpState (Team B) sang Format Dict (Team A)
    để tái sử dụng hàm export_solution_to_csv_and_map.
    """
    @staticmethod
    def adapt(solution: RvrpState, data: ProblemData, date_str: str):
        # 1. Mock Manager & Routing & Solution objects 
        # Vì hàm export nhóm A dùng OR-Tools object, ta phải giả lập behavior của nó
        # Hoặc tốt hơn: Viết lại hàm export để nhận dict thuần. 
        # Ở đây em sẽ REWRITE lại hàm export_solution để nhận Data thuần (Clean Architecture).
        # NHƯNG để nhanh, em sẽ map dữ liệu vào cấu trúc dict mà hàm export CẦN.
        
        # Mapping Problem Data
        start_time_offset = data.time_windows[0][0] # Depot open time
        
        export_data = {
            'num_vehicles': len(solution.routes),
            'vehicle_names': [r.vehicle_type.name for r in solution.routes],
            'vehicle_capacities_kgm': [r.vehicle_type.capacity_kg for r in solution.routes],
            'vehicle_capacities_cbm': [r.vehicle_type.capacity_cbm for r in solution.routes],
            'vehicle_fixed_costs': [r.vehicle_type.fixed_cost for r in solution.routes],
            'vehicle_costs_per_km': [r.vehicle_type.cost_per_km for r in solution.routes],
            'vehicle_costs_per_h': [r.vehicle_type.cost_per_hour for r in solution.routes],
            
            'locations': data.node_ids,
            'node_coordinates': data.coords,
            'demands_kgm': data.demands_kg,
            'demands_cbm': data.demands_cbm,
            'time_windows': data.time_windows,
            'dwell_times': data.service_times,
            'allowed_vehicles': data.allowed_vehicles,
            
            'distance_matrix': data.dist_matrix, # Meter
            # Time matrix export cần xử lý vì ta có Super Matrix (3D)
            # Hàm export nhóm A dùng time_matrix 2D. Ta sẽ lấy time matrix của xe đầu tiên trong route?
            # Hack: Ta sẽ truyền time_matrix của từng xe vào lúc tính toán trong vòng lặp export
        }
        
        return export_data

# --- 4. INFERENCE LOOP ---

def run_inference_pipeline():
    # 1. Split Data
    date_files = split_orders_by_date(path_cfg.ORDER_PATH, path_cfg.TEMP_DATA_DIR)
    
    # 2. Warmup
    # warmup_distance_cache(path_cfg.ORDER_PATH)
    master_depot_id = warmup_and_get_master_id(path_cfg.ORDER_PATH)
    
    # 3. Load Model
    print(f">>> [3/4] Loading Model from {path_cfg.INFERENCE_MODEL_PATH}...")
    if not os.path.exists(path_cfg.INFERENCE_MODEL_PATH):
        print("❌ Error: Model file not found. Please train first or check path in config.")
        return

    model = MaskablePPO.load(path_cfg.INFERENCE_MODEL_PATH, device=ppo_cfg.device)
    
    # 4. Loop Days
    print(f">>> [4/4] Starting Optimization Loop for {len(date_files)} days...")
    
    # Hàm helper cho mask
    def mask_fn(env): return env.valid_action_mask()
    
    for date_str, csv_path in date_files:
        print(f"\n  ► Processing Date: {date_str}")
        try:
            # A. Init Env
            #ISSUE HERE: WHEN INIT ENVIRONMENT, THERE'S NO WAY TO TELL IT WHICH DIST MATRIX TO USE
            env = RVRPEnvironment(
                order_csv_path=csv_path,
                truck_csv_path=path_cfg.TRUCK_PATH,
                is_test_mode=True,
                override_depot_id=master_depot_id 
            )
            env = ActionMasker(env, mask_fn)
            
            # B. Optimize
            obs, _ = env.reset()
            done = False
            
            # (Optional) Tăng số step test để ALNS chạy kỹ hơn
            # Trong env.reset() nó set iters=0. Loop dưới này chạy đến khi Env trả về Done
            # Env Done khi đạt MAX_ITERATIONS (config) hoặc Stop Threshold.
            
            while not done:
                action_masks = get_action_masks(env)
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            # C. Retrieve Result
            real_env = env.unwrapped
            best_sol = real_env.best_solution
            problem_data = real_env.problem_data
            
            print(f"    ✔ Done. Best Cost: {best_sol.objective():,.0f} | Routes: {len(best_sol.routes)} | Unassigned: {len(best_sol.unassigned)}")
            
            # D. Export (Using Team A's Logic)
            # Vì hàm export_solution_to_csv_and_map của nhóm A dính chặt với OR-Tools (nhận tham số 'manager', 'routing'...)
            # Nên ta không gọi trực tiếp được.
            # Ta cần viết 1 hàm 'export_from_rvrp_state' tương đương, dùng pandas/folium thuần.
            
            export_rvrp_result(best_sol, problem_data, date_str, path_cfg.FINAL_REPORT_DIR)
            
        except Exception as e:
            print(f"    ❌ Failed processing {date_str}: {e}")
            import traceback
            traceback.print_exc()

# --- 5. EXPORT FUNCTION (RE-IMPLEMENTED FOR RVRP) ---

# file: inference.py

def export_rvrp_result(solution: RvrpState, data: ProblemData, date_str: str, output_dir: str):
    """
    Xuất kết quả tối ưu ra file Excel (Báo cáo) và HTML Map (Trực quan hóa).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary_rows = []
    detail_rows = []
    
    # Depot Open Time (phút)
    depot_open_time = data.time_windows[0][0]
    
    # --- 1. DATA PROCESSING FOR EXCEL ---
    for r_idx, route in enumerate(solution.routes):
        v_type = route.vehicle_type
        route_id = f"R_{date_str}_{r_idx}"
        
        # Init Metrics
        current_time = depot_open_time
        prev_node = 0 # Start at Depot
        
        # Buffer để tính Load On Board (vì cần tính ngược từ tổng load)
        route_stops_buffer = []
        
        # --- Loop qua các điểm dừng ---
        for seq, node_idx in enumerate(route.node_sequence):
            # Travel Time & Dist
            dist_m = data.dist_matrix[prev_node, node_idx]
            time_m = data.get_travel_time(prev_node, node_idx, v_type.type_id)
            
            current_time += time_m
            arrival_time = current_time
            
            # Time Window Logic
            tw_start, tw_end = data.time_windows[node_idx]
            wait_time = 0
            if current_time < tw_start:
                wait_time = tw_start - current_time
                current_time = tw_start
            
            # Service Time
            service_m = data.service_times[node_idx]
            current_time += service_m
            departure_time = current_time
            
            # Format Time String (HH:MM)
            def fmt_time(t): return f"{int(t//60):02d}:{int(t%60):02d}"
            
            # Append Detail Row
            route_stops_buffer.append({
                "RouteID": route_id,
                "VehicleType": v_type.name,
                "StopOrder": seq + 1,
                "LocationID": data.node_ids[node_idx],
                "Allowed_Trucks": "Check Master", # Placeholder
                "ETA": fmt_time(arrival_time),
                "ETD": fmt_time(departure_time),
                "TimeWindow": f"{fmt_time(tw_start)}-{fmt_time(tw_end)}",
                "Demand_KGM": data.demands_kg[node_idx],
                "Demand_CBM": data.demands_cbm[node_idx],
                "Segment_Dist_KM": round(dist_m / 1000.0, 2),
                "Wait_Min": round(wait_time, 1)
            })
            
            prev_node = node_idx

        # --- Calculate Load On Board (Reverse) ---
        # Logic: Xe xuất phát đầy tải -> đi qua mỗi điểm thì nhẹ đi
        current_load_kg = route.total_load_kg
        current_load_cbm = route.total_load_cbm
        
        for stop in route_stops_buffer:
            stop["Load_On_Board_KG"] = current_load_kg
            stop["Load_On_Board_CBM"] = current_load_cbm
            
            # Trừ hàng đã giao
            current_load_kg -= stop["Demand_KGM"]
            current_load_cbm -= stop["Demand_CBM"]
            
            detail_rows.append(stop)

        # --- Summary Row ---
        summary_rows.append({
            "RouteID": route_id,
            "VehicleID": f"{v_type.name}_{r_idx}",
            "VehicleType": v_type.name,
            "Num_Stops": len(route.node_sequence),
            "Total_Distance_KM": round(route.total_dist_meters / 1000.0, 2),
            "Total_Duration_Min": round(route.total_duration_min, 1),
            "Total_Load_KGM": route.total_load_kg,
            "Capacity_KGM": v_type.capacity_kg,
            "Util_KGM_%": round(route.capacity_utilization * 100, 1),
            "Total_Trip_Cost": round(route.cost, 0)
        })

    # --- 2. EXPORT EXCEL ---
    excel_name = os.path.join(output_dir, f"solution_{date_str}.xlsx")
    try:
        with pd.ExcelWriter(excel_name) as writer:
            if summary_rows:
                pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            else:
                pd.DataFrame(["No Routes"]).to_excel(writer, sheet_name="Summary")
                
            if detail_rows:
                pd.DataFrame(detail_rows).to_excel(writer, sheet_name="Details", index=False)
        print(f"    -> Exported Excel: {excel_name}")
    except Exception as e:
        print(f"    [Warning] Excel export failed: {e}")

    # --- 3. MAP VISUALIZATION (USING RouteVisualizer) ---
    try:
        viz = RouteVisualizer(output_dir=output_dir)
        map_filename = f"map_{date_str}.html"
        # Hàm này sẽ tự động check cache geometry và vẽ OSRM polyline
        viz.visualize_solution(solution, data, filename=map_filename)
    except Exception as e:
        print(f"    [Warning] Map visualization failed: {e}")
    

if __name__ == "__main__":
    run_inference_pipeline()