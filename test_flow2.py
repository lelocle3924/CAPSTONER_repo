# file: test_flow.py

import sys
import os
import numpy as np

# Import Core Modules
from core.real_data_loader import RealDataLoader
from core.data_structures import RvrpState
import alns as vrp
from alns.alns4ppo import ALNS4PPO

# Import Operators cụ thể
from alns.destroyopt import (
    create_random_customer_removal_operator, 
    create_random_route_removal_operator
)
from alns.repairopt import (
    create_greedy_repair_operator, 
    create_regret_repair_operator
)

def print_separator(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_state_summary(tag: str, state: RvrpState):
    """Hàm helper in ra snapshot của State"""
    total_cost = state.objective()
    n_routes = len(state.routes)
    n_unassigned = len(state.unassigned)
    avg_util = state.mean_capacity_utilization * 100
    
    print(f"[{tag}]")
    print(f"  - Cost: {total_cost:,.2f}")
    print(f"  - Routes: {n_routes} | Unassigned: {n_unassigned}")
    print(f"  - Mean Util: {avg_util:.2f}%")
    if n_unassigned > 0:
        print(f"  - Unassigned IDs: {state.unassigned}")

def main():
    # --- 1. SETUP ---
    print_separator("STEP 1: INITIALIZATION")
    
    # Load Data
    loader = RealDataLoader()
    # Đường dẫn file giả định (thầy check lại path nhé)
    order_path = "inputs/CleanData/Split_TransportOrder_1day.csv"
    truck_path = "inputs/MasterData/TruckMaster.csv"
    
    if not os.path.exists(order_path):
        print(f"❌ Input file not found: {order_path}")
        return

    problem_data = loader.load_day_data(order_path, truck_path)
    print(f"✅ Data Loaded. Nodes: {problem_data.num_nodes}. Fleet Types: {len(problem_data.vehicle_types)}")

    # Init Solution
    init_sol = vrp.initial.one_for_one(problem_data)
    print_state_summary("INITIAL SOLUTION", init_sol)

    # Setup ALNS Orchestrator
    rng = np.random.default_rng(2025) # Fixed seed for reproducibility
    alns = ALNS4PPO(rng=rng)

    # --- 2. REGISTER OPERATORS ---
    print_separator("STEP 2: REGISTERING OPERATORS")
    
    # Destroy Ops
    # Index 0: Random Customer Removal (15%)
    alns.add_destroy_operator(create_random_customer_removal_operator(problem_data, ratio=0.15), name="RandomRemoval_15")
    # Index 1: Route Removal (10%)
    alns.add_destroy_operator(create_random_route_removal_operator(problem_data, ratio=0.1), name="RouteRemoval_10")
    
    # Repair Ops
    # Index 0: Greedy
    alns.add_repair_operator(create_greedy_repair_operator(problem_data), name="GreedyRepair")
    # Index 1: Regret-2
    alns.add_repair_operator(create_regret_repair_operator(problem_data, k=2), name="RegretRepair_2")
    
    print("✅ Operators Registered:")
    print("  Destroy:", [op[0] for op in alns.destroy_operators])
    print("  Repair :", [op[0] for op in alns.repair_operators])

    # --- 3. EXECUTE ITERATION 1 ---
    # Scenario: (Random Removal, Greedy Insertion, Accept=1)
    # --- 3. EXECUTE ITERATION 1 ---
    # Scenario: Random Removal + Greedy Repair + Accept Mode = 1 (Explore)
    print_separator("STEP 3: ITERATION 1 (Explore Mode)")
    
    d_idx = 0 
    r_idx = 0 
    accept_mode = 1 # Kể cả tệ hơn cũng lấy
    
    baseline = init_sol
    
    print(f"ACTION: Destroy[{d_idx}] -> Repair[{r_idx}] | Mode: EXPLORE (1)")
    
    # Gọi hàm iterate mới
    baseline_ref, new_sol = alns.iterate(baseline, baseline, d_idx, r_idx, accept_mode, data=problem_data)
    
    print_state_summary("NEW SOLUTION", new_sol)
    
    # Check logic
    if new_sol.objective() > baseline.objective():
        print("-> Result: Worsening ACCEPTED (due to Explore Mode).")
    elif new_sol.objective() < baseline.objective():
        print("-> Result: Improvement Found.")
    else:
        print("-> Result: No Change.")

    # --- 4. EXECUTE ITERATION 2 ---
    # Scenario: Route Removal + Regret Repair + Accept Mode = 0 (Greedy)
    print_separator("STEP 4: ITERATION 2 (Greedy Mode)")
    
    d_idx = 1
    r_idx = 1
    accept_mode = 0 # Chỉ lấy nếu tốt hơn
    
    baseline_2 = new_sol # Dùng kết quả vòng trước làm đầu vào
    
    print(f"ACTION: Destroy[{d_idx}] -> Repair[{r_idx}] | Mode: GREEDY (0)")
    
    baseline_ref_2, new_sol_2 = alns.iterate(baseline_2, baseline_2, d_idx, r_idx, accept_mode, data=problem_data)
    
    print_state_summary("NEW SOLUTION", new_sol_2)
    
    if new_sol_2.objective() > baseline_2.objective():
        # Logic này sẽ không bao giờ xảy ra nếu code đúng
        print("❌ BUG: Worsening accepted in Greedy Mode!") 
    elif new_sol_2.objective() == baseline_2.objective():
        print("-> Result: Worsening REJECTED (kept old solution).")
    else:
        print("-> Result: Improvement Found and Accepted.")

    # --- 5. FEASIBILITY CHECK ---
    print_separator("STEP 5: FINAL INTEGRITY CHECK")
    
    final_routes = new_sol_2.routes

    for i, route in enumerate(final_routes):
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


    # Check 1: Khách hàng có bị rơi rớt không?
    total_served = sum(len(r.node_sequence) for r in final_routes)
    total_unassigned = len(new_sol_2.unassigned)
    total_nodes = problem_data.num_nodes - 1
    
    print(f"Total Nodes: {total_nodes}")
    print(f"Served + Unassigned: {total_served} + {total_unassigned} = {total_served + total_unassigned}")
    
    if total_served + total_unassigned == total_nodes:
        print("✅ CONSERVATION OF MASS: Passed.")
    else:
        print("❌ CRITICAL: Customer count mismatch!")

    # Check 2: Route Feasibility
    all_feasible = True
    for i, r in enumerate(final_routes):
        # Chúng ta tin tưởng vào metrics đã update trong repairopt, nhưng check lại logic cơ bản
        if r.total_load_kg > r.vehicle_type.capacity_kg:
            print(f"❌ Route {i} Overload: {r.total_load_kg} > {r.vehicle_type.capacity_kg}")
            all_feasible = False
            
    if all_feasible:
        print("✅ CAPACITY CONSTRAINT: All routes feasible.")

if __name__ == "__main__":
    main()