import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any
from core.data_structures import RvrpState, Route, ProblemData, VehicleType

# ============================================================
# 1. FEASIBILITY ENGINE & COST CALCULATOR
# ============================================================

def check_sequence_feasibility(data: ProblemData, vehicle: VehicleType, node_sequence: List[int]) -> Tuple[bool, Optional[Dict[str, float]], str]:
    """
    Core feasibility check. Kiểm tra ràng buộc với loại xe 'vehicle' cụ thể.
    """
    # 1. PREFERENCE CHECK (Fail fast)
    for node in node_sequence:
        if vehicle.type_id not in data.allowed_vehicles[node]:
            return False, None, f"Preference Violation at Node {node}"

    # 2. CAPACITY CHECK (Fail fast)
    total_kg = 0.0
    total_cbm = 0.0
    for node in node_sequence:
        total_kg += data.demands_kg[node]
        total_cbm += data.demands_cbm[node]
        
    if total_kg > vehicle.capacity_kg:
        return False, None, f"Capacity KG: {total_kg} > {vehicle.capacity_kg}"
    if total_cbm > vehicle.capacity_cbm:
        return False, None, f"Capacity CBM: {total_cbm} > {vehicle.capacity_cbm}"

    # 3. TIME SIMULATION
    current_time = data.time_windows[0][0] # Depot Open
    total_dist = 0.0
    total_wait = 0.0
    prev_node = 0 
    v_id = vehicle.type_id
    
    for node in node_sequence:
        # Travel
        dist = data.dist_matrix[prev_node, node]
        t_travel = data.get_travel_time(prev_node, node, v_id)
        
        total_dist += dist
        current_time += t_travel
        
        # Time Window
        start_window = data.time_windows[node][0]
        end_window = data.time_windows[node][1]
        
        if current_time > end_window:
            return False, None, f"TW Violation Node {node}"
        
        if current_time < start_window:
            wait = start_window - current_time
            total_wait += wait
            current_time = start_window
            
        # Service
        current_time += data.service_times[node]
        prev_node = node
        
    # Return to Depot
    dist_back = data.dist_matrix[prev_node, 0]
    t_back = data.get_travel_time(prev_node, 0, v_id)
    
    total_dist += dist_back
    current_time += t_back
    
    # Check Depot Limits
    if current_time > data.time_windows[0][1]:
        return False, None, "Depot Closing Violation"
        
    total_duration = current_time - data.time_windows[0][0]
    if total_duration > data.max_route_duration:
        return False, None, "Max Duration Violation"

    metrics = {
        "total_dist_meters": total_dist,
        "total_duration_min": total_duration,
        "total_wait_time_min": total_wait,
        "total_load_kg": total_kg,
        "total_load_cbm": total_cbm
    }
    return True, metrics, "OK"


def calculate_insertion_cost(data: ProblemData, route: Route, customer: int, idx: int) -> Tuple[float, Optional[Dict[str, float]], Optional[VehicleType]]:
    """
    Tính chi phí chèn customer.
    [DYNAMIC FLEET]: Tự động thử xe lớn hơn nếu xe hiện tại không chở nổi.
    Returns: (Absolute_Cost, Metrics, Selected_Vehicle_Type)
    """
    new_sequence = route.node_sequence[:]
    new_sequence.insert(idx, customer)
    
    current_vehicle = route.vehicle_type
    
    # 1. Thử xe hiện tại
    is_feasible, metrics, _ = check_sequence_feasibility(data, current_vehicle, new_sequence)
    if is_feasible:
        dist_km = metrics["total_dist_meters"] / 1000.0
        dur_hr = metrics["total_duration_min"] / 60.0
        cost = current_vehicle.fixed_cost + (dist_km * current_vehicle.cost_per_km) + (dur_hr * current_vehicle.cost_per_hour)
        return cost, metrics, current_vehicle

    # 2. Thử Upgrade (chỉ khi xe hiện tại fail)
    # Sort fleet theo capacity tăng dần
    sorted_fleet = sorted(data.vehicle_types, key=lambda v: v.capacity_kg)
    
    for v in sorted_fleet:
        # Bỏ qua xe nhỏ hơn hoặc bằng xe hiện tại (đã fail ở bước 1)
        if v.capacity_kg <= current_vehicle.capacity_kg: 
            continue
            
        is_feasible, metrics, _ = check_sequence_feasibility(data, v, new_sequence)
        if is_feasible:
            # Found upgrade!
            dist_km = metrics["total_dist_meters"] / 1000.0
            dur_hr = metrics["total_duration_min"] / 60.0
            cost = v.fixed_cost + (dist_km * v.cost_per_km) + (dur_hr * v.cost_per_hour)
            return cost, metrics, v
            
    return float('inf'), None, None


def _try_create_new_route(data: ProblemData, customer: int) -> Optional[Route]:
    """Tìm xe rẻ nhất cho route mới"""
    sorted_fleet = sorted(data.vehicle_types, key=lambda x: x.fixed_cost)
    for v in sorted_fleet:
        is_feasible, metrics, _ = check_sequence_feasibility(data, v, [customer])
        if is_feasible:
            return Route(v, [customer], 
                         total_dist_meters=metrics["total_dist_meters"],
                         total_duration_min=metrics["total_duration_min"],
                         total_wait_time_min=metrics["total_wait_time_min"],
                         total_load_kg=metrics["total_load_kg"],
                         total_load_cbm=metrics["total_load_cbm"])
    return None

# ============================================================
# 3. REPAIR OPERATORS
# ============================================================

def create_greedy_repair_operator(data: ProblemData):
    def greedy_repair(state: RvrpState, rng, data: ProblemData):
        rng.shuffle(state.unassigned)
        while state.unassigned:
            customer = state.unassigned.pop()
            best_diff = float('inf')
            best_r_idx = -1
            best_pos = -1
            best_metrics = None
            best_vehicle = None 
            
            # 1. Try insert into existing routes
            for r_idx, route in enumerate(state.routes):
                prev_cost = route.cost 
                
                for i in range(len(route.node_sequence) + 1):
                    new_abs_cost, metrics, new_veh = calculate_insertion_cost(data, route, customer, i)
                    
                    if metrics:
                        diff = new_abs_cost - prev_cost
                        if diff < best_diff:
                            best_diff = diff
                            best_r_idx = r_idx
                            best_pos = i
                            best_metrics = metrics
                            best_vehicle = new_veh
                            
            # 2. Try create new route
            new_route = _try_create_new_route(data, customer)
            if new_route:
                new_cost = new_route.cost
                if new_cost < best_diff:
                    best_diff = new_cost
                    best_r_idx = -1 
            
            # 3. Apply Best Move
            if best_r_idx != -1:
                target_route = state.routes[best_r_idx]
                target_route.node_sequence.insert(best_pos, customer)
                # Update attributes
                target_route.total_dist_meters = best_metrics["total_dist_meters"]
                target_route.total_duration_min = best_metrics["total_duration_min"]
                target_route.total_wait_time_min = best_metrics["total_wait_time_min"]
                target_route.total_load_kg = best_metrics["total_load_kg"]
                target_route.total_load_cbm = best_metrics["total_load_cbm"]
                
                # [CRITICAL] Update Vehicle Type if Upgraded
                if best_vehicle.type_id != target_route.vehicle_type.type_id:
                    target_route.vehicle_type = best_vehicle
                    
            elif new_route:
                state.routes.append(new_route)
            else:
                pass # Infeasible to serve this customer -> Leave unassigned
                
        return state
    return greedy_repair


def create_criticality_repair_operator(data: ProblemData):
    """
    Repair dựa trên độ ưu tiên (Criticality).
    Logic UPGRADE VEHICLE đã được tích hợp.
    """
    depot_dists = data.dist_matrix[0, :]
    min_tw = 30.0 
    
    def get_importance(node_idx):
        norm_dem = data.demands_kg[node_idx] / 1000.0
        norm_dist = depot_dists[node_idx] / 10000.0
        tw_len = data.time_windows[node_idx][1] - data.time_windows[node_idx][0]
        norm_tw = 1.0 / (max(tw_len, min_tw) / 60.0)
        return norm_dem + norm_dist + norm_tw

    def criticality_repair(state: RvrpState, rng, data: ProblemData):
        # Sort unassigned: Khó nhất xếp cuối để pop ra đầu
        state.unassigned.sort(key=get_importance) 

        while state.unassigned:
            customer = state.unassigned.pop() 
            
            best_diff = float('inf')
            best_r_idx = -1
            best_pos = -1
            best_metrics = None
            best_vehicle = None
            
            # 1. Existing Routes
            for r_idx, route in enumerate(state.routes):
                prev_cost = route.cost 
                
                for i in range(len(route.node_sequence) + 1):
                    new_abs_cost, metrics, new_veh = calculate_insertion_cost(data, route, customer, i)
                    if metrics:
                        diff = new_abs_cost - prev_cost
                        if diff < best_diff:
                            best_diff = diff
                            best_r_idx = r_idx
                            best_pos = i
                            best_metrics = metrics
                            best_vehicle = new_veh
                            
            # 2. New Route
            new_route = _try_create_new_route(data, customer)
            if new_route:
                if new_route.cost < best_diff:
                    best_diff = new_route.cost
                    best_r_idx = -1
                    
            # 3. Apply
            if best_r_idx != -1:
                target_route = state.routes[best_r_idx]
                target_route.node_sequence.insert(best_pos, customer)
                target_route.total_dist_meters = best_metrics["total_dist_meters"]
                target_route.total_duration_min = best_metrics["total_duration_min"]
                target_route.total_wait_time_min = best_metrics["total_wait_time_min"]
                target_route.total_load_kg = best_metrics["total_load_kg"]
                target_route.total_load_cbm = best_metrics["total_load_cbm"]
                
                # [CRITICAL] Update Vehicle
                if best_vehicle.type_id != target_route.vehicle_type.type_id:
                    target_route.vehicle_type = best_vehicle
            elif new_route:
                state.routes.append(new_route)

        return state
    return criticality_repair


def create_regret_repair_operator(data: ProblemData, k: int = 2):
    """
    Regret Repair: Tính nuối tiếc nếu không chọn lựa chọn tốt nhất.
    Logic UPGRADE VEHICLE đã được tích hợp.
    """
    def regret_repair(state: RvrpState, rng, data: ProblemData):
        while state.unassigned:
            # List chứa thông tin regret của từng customer
            # (regret_val, customer, r_idx, pos, metrics, vehicle)
            candidates_regret = []
            
            for customer in state.unassigned:
                # Tìm tất cả feasible insertions cho customer này
                valid_insertions = [] # (cost_increase, r_idx, pos, metrics, vehicle)
                
                # 1. Quét Existing Routes
                for r_idx, route in enumerate(state.routes):
                    prev_cost = route.cost
                    
                    local_best_diff = float('inf')
                    local_best_entry = None # (diff, pos, metrics, veh)
                    
                    for i in range(len(route.node_sequence) + 1):
                        new_cost, metrics, new_veh = calculate_insertion_cost(data, route, customer, i)
                        if metrics:
                            diff = new_cost - prev_cost
                            if diff < local_best_diff:
                                local_best_diff = diff
                                local_best_entry = (diff, i, metrics, new_veh)
                    
                    if local_best_entry:
                        valid_insertions.append((local_best_entry[0], r_idx, local_best_entry[1], local_best_entry[2], local_best_entry[3]))
                
                # 2. Quét New Route Option
                new_route = _try_create_new_route(data, customer)
                if new_route:
                    # New route metrics
                    m = {
                        "total_dist_meters": new_route.total_dist_meters,
                        "total_duration_min": new_route.total_duration_min,
                        "total_wait_time_min": new_route.total_wait_time_min,
                        "total_load_kg": new_route.total_load_kg,
                        "total_load_cbm": new_route.total_load_cbm
                    }
                    valid_insertions.append((new_route.cost, -1, 0, m, new_route.vehicle_type))
                
                # 3. Tính Regret
                if not valid_insertions: continue # Khách này ko chèn được đâu cả
                
                valid_insertions.sort(key=lambda x: x[0]) # Sort theo cost tăng dần
                
                best = valid_insertions[0]
                if len(valid_insertions) >= k:
                    second = valid_insertions[k-1]
                    regret_val = second[0] - best[0]
                else:
                    regret_val = float('inf') # Must insert now
                    
                candidates_regret.append({
                    "regret": regret_val,
                    "customer": customer,
                    "r_idx": best[1],
                    "pos": best[2],
                    "metrics": best[3],
                    "vehicle": best[4]
                })
            
            if not candidates_regret: break
            
            # Chọn Max Regret
            candidates_regret.sort(key=lambda x: x["regret"], reverse=True)
            winner = candidates_regret[0]
            
            # Apply
            cust = winner["customer"]
            state.unassigned.remove(cust)
            
            if winner["r_idx"] != -1:
                target_route = state.routes[winner["r_idx"]]
                target_route.node_sequence.insert(winner["pos"], cust)
                m = winner["metrics"]
                target_route.total_dist_meters = m["total_dist_meters"]
                target_route.total_duration_min = m["total_duration_min"]
                target_route.total_wait_time_min = m["total_wait_time_min"]
                target_route.total_load_kg = m["total_load_kg"]
                target_route.total_load_cbm = m["total_load_cbm"]
                
                # [CRITICAL] Upgrade Vehicle
                best_veh = winner["vehicle"]
                if best_veh.type_id != target_route.vehicle_type.type_id:
                    target_route.vehicle_type = best_veh
            else:
                new_route = _try_create_new_route(data, cust) # Re-create object
                if new_route: state.routes.append(new_route)

        return state
    return regret_repair