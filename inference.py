# file: inference.py

import pandas as pd
import numpy as np
import os
import shutil
import glob
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

# Core Imports
from core.data_structures import RvrpState, ProblemData
from core.real_data_loader import RealDataLoader
from core.visualizer import RouteVisualizer
from config import PathConfig, PPOConfig
from ppo.rvrpenv import RVRPEnvironment
from core.distance_matrix import DistanceMatrixCalculator

# [CRITICAL] Re-using Export Logic from Team A (Adapted)
# Giả sử file của nhóm A tên là export_solution.py nằm ở root hoặc thư mục utils
# Nếu chưa có, copy nội dung file export_solution.py vào utils/export_solution.py
# Ở đây em sẽ viết Adapter để gọi nó.
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

def warmup_distance_cache(full_order_path):
    """
    Chạy DistanceMatrixCalculator trên toàn bộ tập khách hàng 1 lần
    để cache hết khoảng cách, tránh gọi API lẻ tẻ trong vòng lặp.
    """
    print(">>> [2/4] Warming up Distance Matrix Cache...")
    # Logic: Calculator tự động check cache và tải thiếu.
    # Chỉ cần init nó với file tổng là được.
    calc = DistanceMatrixCalculator(full_order_path, path_cfg.TRUCK_PATH)
    calc.calculate_matrices() # Trigger download & save
    print("    Cache warm-up complete.")

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

def export_rvrp_result(solution: RvrpState, data: ProblemData, date_str: str, output_dir: str):
    """
    Phiên bản Export tương thích với RvrpState của Team B.
    Output giống hệt file của nhóm A.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary_rows = []
    detail_rows = []
    
    # Import map utils
    import folium
    import polyline
    import requests
    import random
    
    def get_color(n): 
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
    
    # Map Setup
    depot_coord = data.coords[0]
    m = folium.Map(location=depot_coord, zoom_start=11, tiles='CartoDB positron')
    folium.Marker(depot_coord, icon=folium.Icon(color='red', icon='home'), popup="Depot").add_to(m)
    
    for r_idx, route in enumerate(solution.routes):
        v_type = route.vehicle_type
        
        # Route Metrics Calculation (Recalculate for display accuracy)
        current_time = data.time_windows[0][0] # Start at Depot Open
        current_load_kg = 0
        current_load_cbm = 0
        
        path_coords = [tuple(data.coords[0])] # Start Depot
        prev_node = 0
        
        route_detail_buffer = []
        
        # --- Iterate Nodes ---
        for seq, node_idx in enumerate(route.node_sequence):
            # Travel
            dist = data.dist_matrix[prev_node, node_idx]
            time_travel = data.get_travel_time(prev_node, node_idx, v_type.type_id)
            current_time += time_travel
            
            # Arrival & Wait
            eta = current_time
            start_win = data.time_windows[node_idx][0]
            if current_time < start_win:
                current_time = start_win
            
            # Service
            current_time += data.service_times[node_idx]
            
            # Load
            dem_kg = data.demands_kg[node_idx]
            dem_cbm = data.demands_cbm[node_idx]
            current_load_kg += dem_kg
            current_load_cbm += dem_cbm
            
            # Add to map path
            coord = tuple(data.coords[node_idx])
            path_coords.append(coord)
            
            # Add Detail Row
            route_detail_buffer.append({
                "RouteID": f"R_{date_str}_{r_idx}",
                "VehicleType": v_type.name,
                "StopOrder": seq + 1,
                "LocationID": data.node_ids[node_idx],
                "Allowed_Trucks": "Check Master", # Placeholder
                "ETA": f"{int(eta//60):02d}:{int(eta%60):02d}",
                "TimeWindow": f"{int(start_win//60):02d}:{int(start_win%60):02d}",
                "Demand_KGM": dem_kg,
                "Demand_CBM": dem_cbm,
                "Segment_Dist_KM": dist / 1000.0,
                "Segment_Cost": 0 # TODO: Breakdown cost
            })
            
            # Map Marker
            folium.CircleMarker(
                coord, radius=4, color=get_color(1), fill=True, 
                popup=f"{data.node_ids[node_idx]} (Seq {seq+1})"
            ).add_to(m)
            
            prev_node = node_idx
            
        # --- Return Depot ---
        path_coords.append(tuple(data.coords[0]))
        
        # Draw PolyLine (OSRM or Straight)
        # Simple Straight Line for robustness now
        folium.PolyLine(path_coords, color=get_color(1), weight=2.5, opacity=0.8).add_to(m)
        
        # Summary Row
        summary_rows.append({
            "RouteID": f"R_{date_str}_{r_idx}",
            "VehicleID": f"{v_type.name}_{r_idx}",
            "VehicleType": v_type.name,
            "Num_Stops": len(route.node_sequence),
            "Total_Distance_KM": route.total_dist_meters / 1000.0,
            "Total_Load_KGM": route.total_load_kg,
            "Util_KGM_%": route.capacity_utilization * 100,
            "Total_Trip_Cost": route.cost
        })
        
        # Add Load On Board info (Reverse calculation)
        # Logic: Load on board giảm dần khi đi qua các điểm
        load_on_board = route.total_load_kg
        for row in route_detail_buffer:
            row['Load_On_Board'] = load_on_board
            load_on_board -= row['Demand_KGM']
            detail_rows.append(row)

    # Save CSV
    excel_name = os.path.join(output_dir, f"solution_{date_str}.xlsx")
    with pd.ExcelWriter(excel_name) as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(detail_rows).to_excel(writer, sheet_name="Details", index=False)
        
    # Save Map
    map_name = os.path.join(output_dir, f"map_{date_str}.html")
    m.save(map_name)
    print(f"    -> Exported: {excel_name} & {map_name}")

if __name__ == "__main__":
    run_inference_pipeline()