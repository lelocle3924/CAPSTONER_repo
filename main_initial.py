# --- START OF FILE main_initial.py ---

import os
import sys
import pandas as pd
import numpy as np

# Thêm đường dẫn để python tìm thấy các module core và alns
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alns.vrp4ppo import initial
from core.real_data_loader import RealDataLoader
from core.data_structures import ProblemData, RvrpState

def main():
    # --- 1. CONFIG PATHS ---
    # Đảm bảo đường dẫn trỏ đúng tới nơi chứa file
    order_csv_path = "inputs/CleanData/Split_TransportOrder_1day.csv"
    truck_csv_path = "inputs/MasterData/TruckMaster.csv"

    # Kiểm tra file tồn tại
    if not os.path.exists(order_csv_path) or not os.path.exists(truck_csv_path):
        print(f"❌ Error: Input files not found.")
        print(f"  - Order: {os.path.exists(order_csv_path)} ({order_csv_path})")
        print(f"  - Truck: {os.path.exists(truck_csv_path)} ({truck_csv_path})")
        return

    # --- 2. LOAD DATA (PIPELINE TEST) ---
    print("\n" + "="*50)
    print("STEP 1: LOADING DATA & GENERATING MATRICES")
    print("="*50)
    
    loader = RealDataLoader()
    try:
        # Loader bây giờ tự gọi DistanceMatrixCalculator và tự xây dựng Super Time Matrix
        problem_data = loader.load_day_data(order_csv_path, truck_csv_path)
        
        print(f"\n✅ Data Loaded Successfully!")
        print(f"  - Nodes: {problem_data.num_nodes} (1 Depot + {problem_data.num_nodes - 1} Customers)")
        print(f"  - Fleet Types Loaded: {len(problem_data.vehicle_types)}")
        print(f"  - Super Time Matrix Shape: {problem_data.super_time_matrix.shape} (Type, Node, Node)")
        
        # Verify Fleet Loaded
        print("  - Available Vehicle Types:")
        for v in problem_data.vehicle_types:
            print(f"    + [{v.type_id}] {v.name}: Cap={v.capacity_kg}kg, Speed={v.speed_kmh}km/h, Cost/km={v.cost_per_km}")

    except Exception as e:
        print(f"❌ Critical Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 3. GENERATE INITIAL SOLUTION ---
    print("\n" + "="*50)
    print("STEP 2: GENERATING INITIAL SOLUTION (One-for-One)")
    print("="*50)

    try:
        initial_solution = initial.one_for_one(problem_data)
        
        print(f"✅ Initial Solution Generated!")
        print(f"  - Total Objective Cost: {initial_solution.objective():,.2f}")
        print(f"  - Total Routes: {len(initial_solution.routes)}")
        print(f"  - Unassigned Customers: {len(initial_solution.unassigned)}")
        
        # --- 4. DETAILED REPORT (VERIFICATION) ---
        print("\n" + "="*50)
        print("STEP 3: DETAILED ROUTE INSPECTION")
        print("="*50)
        
        if not initial_solution.routes:
            print("⚠️ No routes generated. Check constraints/matrices.")
        
        for i, route in enumerate(initial_solution.routes):
            v_name = route.vehicle_type.name
            cust_id = [problem_data.node_ids[route.node_sequence[i]] for i in range(0, len(route.node_sequence))]
            load_kg = route.total_load_kg
            cap_kg = route.vehicle_type.capacity_kg
            util = route.capacity_utilization * 100
            dist_km = route.total_dist_meters / 1000.0
            dur_min = route.total_duration_min
            cost = route.cost
            
            print(f"Route {i+1:02d} | Truck: {v_name:<5} | Cust: {cust_id} | "
                  f"Load: {load_kg:6.1f}/{cap_kg:<5} kg ({util:5.1f}%) | "
                  f"Dist: {dist_km:5.1f} km | Time: {dur_min:5.0f} min | Cost: {cost:8.2f}")

        # --- 5. FINAL VERDICT ---
        print("-" * 50)
        total_customers = problem_data.num_nodes - 1
        served = len(initial_solution.routes)
        
        if len(initial_solution.unassigned) == 0:
            print(f"🎉 SUCCESS: All {total_customers} customers served.")
            print("   The One-for-One heuristic successfully matched vehicles to demands.")
        else:
            print(f"⚠️ WARNING: {len(initial_solution.unassigned)} customers left unassigned.")
            print(f"   Unassigned Indices: {initial_solution.unassigned}")
            
    except Exception as e:
        print(f"❌ Error during solution generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()