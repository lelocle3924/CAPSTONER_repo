import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any
from numba import njit
from core.data_structures import RvrpState, Route, ProblemData, VehicleType

# ============================================================
# 0. HELPER FUNCTIONS
# ============================================================

@njit(fastmath=True, cache=True)
def _fast_check_feasibility_numba(
    dist_mat: np.ndarray,      # 2D Array
    time_mat: np.ndarray,      # 2D Array (Specific to vehicle speed)
    tw: np.ndarray,            # 2D Array [Start, End]
    service: np.ndarray,       # 1D Array
    dem_kg: np.ndarray,        # 1D Array
    dem_cbm: np.ndarray,       # 1D Array
    cap_kg: float,
    cap_cbm: float,
    max_dur: float,
    depot_open: float,
    depot_close: float,
    sequence: np.ndarray       # 1D Array (Int) - The route sequence excluding depot
):
    """
    Low-level C-speed feasibility check.
    Returns: (Status_Int, dist, dur, wait, kg, cbm)
    Status: 1=OK, -1=CapKG, -2=CapCBM, -3=TW, -4=DepotClose, -5=MaxDur
    """
    
    # 1. Capacity Check (Vectorized-like accumulation)
    total_kg = 0.0
    total_cbm = 0.0
    seq_len = sequence.shape[0]
    
    for i in range(seq_len):
        node = sequence[i]
        total_kg += dem_kg[node]
        total_cbm += dem_cbm[node]
        
    if total_kg > cap_kg: return -1, 0.0, 0.0, 0.0, total_kg, total_cbm
    if total_cbm > cap_cbm: return -2, 0.0, 0.0, 0.0, total_kg, total_cbm

    # 2. Time & Distance Simulation
    curr_time = depot_open
    total_dist = 0.0
    total_wait = 0.0
    prev = 0 # Depot
    
    for i in range(seq_len):
        node = sequence[i]
        
        # Travel
        total_dist += dist_mat[prev, node]
        curr_time += time_mat[prev, node]
        
        # Time Window
        start_w = tw[node, 0]
        end_w = tw[node, 1]
        
        if curr_time > end_w:
            return -3, 0.0, 0.0, 0.0, total_kg, total_cbm # Late arrival
        
        if curr_time < start_w:
            wait = start_w - curr_time
            total_wait += wait
            curr_time = start_w
            
        # Service
        curr_time += service[node]
        prev = node
        
    # Return to Depot
    total_dist += dist_mat[prev, 0]
    curr_time += time_mat[prev, 0]
    
    if curr_time > depot_close:
        return -4, 0.0, 0.0, 0.0, total_kg, total_cbm
        
    total_dur = curr_time - depot_open
    if total_dur > max_dur:
        return -5, 0.0, 0.0, 0.0, total_kg, total_cbm
        
    return 1, total_dist, total_dur, total_wait, total_kg, total_cbm


# ============================================================
# 1. FEASIBILITY ENGINE & COST CALCULATOR
# ============================================================

def check_sequence_feasibility(
    data: ProblemData, 
    vehicle: VehicleType, 
    node_sequence: List[int]
) -> Tuple[bool, Optional[Dict[str, float]], str]:
    
    # 1. PREFERENCE CHECK (Still Python - usually fast enough via Dictionary/Set)
    # Nếu bottleneck, chuyển allowed_vehicles thành bitmask matrix cho Numba
    for node in node_sequence:
        if vehicle.type_id not in data.allowed_vehicles[node]:
            return False, None, f"Preference Violation at Node {node}"

    # 2. CALL NUMBA KERNEL
    # Convert list to numpy array for Numba
    seq_arr = np.array(node_sequence, dtype=np.int32)
    
    # Get specific time matrix slice for this vehicle type
    # data.super_time_matrix shape (NumVeh, N, N)
    time_slice = data.super_time_matrix[vehicle.type_id]
    
    status, dist, dur, wait, kg, cbm = _fast_check_feasibility_numba(
        data.dist_matrix,
        time_slice,
        data.time_windows,
        data.service_times,
        data.demands_kg,
        data.demands_cbm,
        vehicle.capacity_kg,
        vehicle.capacity_cbm,
        data.max_route_duration,
        data.time_windows[0][0],
        data.time_windows[0][1],
        seq_arr
    )
    
    if status != 1:
        reasons = { -1: "CapKG", -2: "CapCBM", -3: "TW", -4: "DepotClose", -5: "MaxDur" }
        return False, None, reasons.get(status, "Unknown")

    metrics = {
        "total_dist_meters": dist,
        "total_duration_min": dur,
        "total_wait_time_min": wait,
        "total_load_kg": kg,
        "total_load_cbm": cbm
    }
    return True, metrics, "OK"


def calculate_insertion_cost(data: ProblemData, route: Route, customer: int, idx: int) -> Tuple[float, Optional[Dict[str, float]], Optional[VehicleType]]:
    """
    Wraps check_sequence_feasibility with Auto-Upgrade logic.
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
    sorted_fleet = sorted(data.vehicle_types, key=lambda v: v.capacity_kg)
    
    for v in sorted_fleet:
        if v.capacity_kg <= current_vehicle.capacity_kg: 
            continue
            
        is_feasible, metrics, _ = check_sequence_feasibility(data, v, new_sequence)
        if is_feasible:
            dist_km = metrics["total_dist_meters"] / 1000.0
            dur_hr = metrics["total_duration_min"] / 60.0
            cost = v.fixed_cost + (dist_km * v.cost_per_km) + (dur_hr * v.cost_per_hour)
            return cost, metrics, v
            
    return float('inf'), None, None

def _get_candidate_routes(data: ProblemData, state: RvrpState, customer_idx: int, limit: int = 50) -> List[Tuple[float, int, Route]]:
    """
    Trả về danh sách Top-K routes gần khách hàng nhất dựa trên Centroid.
    Return Format: List[(distance_sq, route_idx, route_obj)]
    """
    # Nếu số lượng route ít, không cần prune tốn công sort
    if len(state.routes) <= limit:
        # Trả về tất cả, distance = 0 (dummy)
        return [(0.0, idx, r) for idx, r in enumerate(state.routes)]

    cust_lat = data.coords[customer_idx][0]
    cust_lon = data.coords[customer_idx][1]
    
    candidates = []
    for r_idx, route in enumerate(state.routes):
        # Euclidean approximation (đủ nhanh và chính xác cho việc pruning)
        d_lat = route.centroid_lat - cust_lat
        d_lon = route.centroid_lon - cust_lon
        dist_sq = d_lat*d_lat + d_lon*d_lon
        candidates.append((dist_sq, r_idx, route))
    
    # Partial Sort: Chỉ lấy Top K phần tử nhỏ nhất (nhanh hơn sort full)
    # Nếu limit nhỏ so với len, dùng heapq.nsmallest hoặc argpartition sẽ nhanh hơn.
    # Nhưng python list.sort() rất tối ưu (Timsort), với N<1000 thì sort full cũng ok.
    candidates.sort(key=lambda x: x[0])
    
    return candidates[:limit]

def _try_create_new_route(data: ProblemData, customer: int) -> Optional[Route]:
    sorted_fleet = sorted(data.vehicle_types, key=lambda x: x.fixed_cost)
    for v in sorted_fleet:
        is_feasible, metrics, _ = check_sequence_feasibility(data, v, [customer])
        if is_feasible:
            r = Route(v, [customer], 
                         total_dist_meters=metrics["total_dist_meters"],
                         total_duration_min=metrics["total_duration_min"],
                         total_wait_time_min=metrics["total_wait_time_min"],
                         total_load_kg=metrics["total_load_kg"],
                         total_load_cbm=metrics["total_load_cbm"])
            r.update_centroid(data) # Calculate centroid once
            return r
    return None

# ============================================================
# 3. REPAIR OPERATORS
# ============================================================
def create_greedy_repair_operator(data: ProblemData):
    MAX_ROUTES = 50 

    def greedy_repair(state: RvrpState, rng, data: ProblemData):
        rng.shuffle(state.unassigned)
        while state.unassigned:
            customer = state.unassigned.pop()
            best_diff = float('inf')
            best_r_idx = -1
            best_pos = -1
            best_metrics = None
            best_vehicle = None 
            
            # --- USE HELPER ---
            top_routes = _get_candidate_routes(data, state, customer, limit=MAX_ROUTES)
            
            # 1. Check Pruned Routes
            for _, r_idx, route in top_routes:
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
                            
            # 2. Check New Route
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
                target_route.update_centroid(data)
                
                if best_vehicle.type_id != target_route.vehicle_type.type_id:
                    target_route.vehicle_type = best_vehicle
            elif new_route:
                state.routes.append(new_route)
            else:
                pass 
                
        return state
    return greedy_repair


def create_criticality_repair_operator(data: ProblemData):
    depot_dists = data.dist_matrix[0, :]
    min_tw = 30.0 
    MAX_ROUTES = 50
    
    def get_importance(node_idx):
        norm_dem = data.demands_kg[node_idx] / 1000.0
        norm_dist = depot_dists[node_idx] / 10000.0
        tw_len = data.time_windows[node_idx][1] - data.time_windows[node_idx][0]
        norm_tw = 1.0 / (max(tw_len, min_tw) / 60.0)
        return norm_dem + norm_dist + norm_tw

    def criticality_repair(state: RvrpState, rng, data: ProblemData):
        state.unassigned.sort(key=get_importance) 

        while state.unassigned:
            customer = state.unassigned.pop() 
            best_diff = float('inf')
            best_r_idx = -1
            best_pos = -1
            best_metrics = None
            best_vehicle = None
            
            # --- USE HELPER ---
            top_routes = _get_candidate_routes(data, state, customer, limit=MAX_ROUTES)
            
            for _, r_idx, route in top_routes:
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
                            
            new_route = _try_create_new_route(data, customer)
            if new_route:
                if new_route.cost < best_diff:
                    best_diff = new_route.cost
                    best_r_idx = -1
                    
            if best_r_idx != -1:
                target_route = state.routes[best_r_idx]
                target_route.node_sequence.insert(best_pos, customer)
                target_route.total_dist_meters = best_metrics["total_dist_meters"]
                target_route.total_duration_min = best_metrics["total_duration_min"]
                target_route.total_wait_time_min = best_metrics["total_wait_time_min"]
                target_route.total_load_kg = best_metrics["total_load_kg"]
                target_route.total_load_cbm = best_metrics["total_load_cbm"]
                target_route.update_centroid(data)
                
                if best_vehicle.type_id != target_route.vehicle_type.type_id:
                    target_route.vehicle_type = best_vehicle
            elif new_route:
                state.routes.append(new_route)

        return state
    return criticality_repair


def create_regret_repair_operator(data: ProblemData, k: int = 2):
    MAX_ROUTES = 30 # Prune chặt hơn vì Regret chạy chậm
    
    def regret_repair(state: RvrpState, rng, data: ProblemData):
        while state.unassigned:
            candidates_regret = []
            
            for customer in state.unassigned:
                valid_insertions = [] 
                
                # --- USE HELPER ---
                top_routes = _get_candidate_routes(data, state, customer, limit=MAX_ROUTES)
                
                # 1. Existing Routes
                for _, r_idx, route in top_routes:
                    prev_cost = route.cost
                    local_best_diff = float('inf')
                    local_best_entry = None 
                    
                    for i in range(len(route.node_sequence) + 1):
                        new_cost, metrics, new_veh = calculate_insertion_cost(data, route, customer, i)
                        if metrics:
                            diff = new_cost - prev_cost
                            if diff < local_best_diff:
                                local_best_diff = diff
                                local_best_entry = (diff, i, metrics, new_veh)
                    
                    if local_best_entry:
                        valid_insertions.append((local_best_entry[0], r_idx, local_best_entry[1], local_best_entry[2], local_best_entry[3]))
                
                # 2. New Route
                new_route = _try_create_new_route(data, customer)
                if new_route:
                    m = {
                        "total_dist_meters": new_route.total_dist_meters,
                        "total_duration_min": new_route.total_duration_min,
                        "total_wait_time_min": new_route.total_wait_time_min,
                        "total_load_kg": new_route.total_load_kg,
                        "total_load_cbm": new_route.total_load_cbm
                    }
                    valid_insertions.append((new_route.cost, -1, 0, m, new_route.vehicle_type))
                
                if not valid_insertions: continue 
                
                valid_insertions.sort(key=lambda x: x[0]) 
                best = valid_insertions[0]
                if len(valid_insertions) >= k:
                    regret_val = valid_insertions[k-1][0] - best[0]
                else:
                    regret_val = float('inf') 
                    
                candidates_regret.append({
                    "regret": regret_val, "customer": customer, "r_idx": best[1],
                    "pos": best[2], "metrics": best[3], "vehicle": best[4]
                })
            
            if not candidates_regret: break
            
            candidates_regret.sort(key=lambda x: x["regret"], reverse=True)
            winner = candidates_regret[0]
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
                target_route.update_centroid(data)
                
                if winner["vehicle"].type_id != target_route.vehicle_type.type_id:
                    target_route.vehicle_type = winner["vehicle"]
            else:
                new_route = _try_create_new_route(data, cust) 
                if new_route: state.routes.append(new_route)

        return state
    return regret_repair


def create_grasp_repair_operator(data: ProblemData, rcl_size: int = 3):
    MAX_ROUTES = 30
    def grasp_repair(state: RvrpState, rng, data: ProblemData):
        rng.shuffle(state.unassigned)
        while state.unassigned:
            all_moves = []
            for customer in state.unassigned:
                # --- USE HELPER ---
                top_routes = _get_candidate_routes(data, state, customer, limit=MAX_ROUTES)
                
                for _, r_idx, route in top_routes:
                    prev_cost = route.cost
                    local_best_diff = float('inf')
                    local_best_entry = None
                    for i in range(len(route.node_sequence) + 1):
                        cost, metrics, veh = calculate_insertion_cost(data, route, customer, i)
                        if metrics:
                            diff = cost - prev_cost
                            if diff < local_best_diff:
                                local_best_diff = diff
                                local_best_entry = (diff, customer, r_idx, i, metrics, veh)
                    if local_best_entry: all_moves.append(local_best_entry)
                            
                new_route = _try_create_new_route(data, customer)
                if new_route:
                    m = {"total_dist_meters": new_route.total_dist_meters, "total_duration_min": new_route.total_duration_min,
                         "total_wait_time_min": new_route.total_wait_time_min, "total_load_kg": new_route.total_load_kg,
                         "total_load_cbm": new_route.total_load_cbm}
                    all_moves.append((new_route.cost, customer, -1, 0, m, new_route.vehicle_type))

            if not all_moves: break 
            
            all_moves.sort(key=lambda x: x[0])
            top_n = min(len(all_moves), rcl_size)
            selected_idx = rng.integers(0, top_n)
            winner = all_moves[selected_idx]
            
            _, cust, r_idx, pos, m, veh = winner
            state.unassigned.remove(cust)
            
            if r_idx != -1:
                target = state.routes[r_idx]
                target.node_sequence.insert(pos, cust)
                target.total_dist_meters = m["total_dist_meters"]
                target.total_duration_min = m["total_duration_min"]
                target.total_wait_time_min = m["total_wait_time_min"]
                target.total_load_kg = m["total_load_kg"]
                target.total_load_cbm = m["total_load_cbm"]
                target.update_centroid(data)
                if target.vehicle_type.type_id != veh.type_id: target.vehicle_type = veh
            else:
                real_new_route = _try_create_new_route(data, cust) 
                if real_new_route: state.routes.append(real_new_route)
        return state
    return grasp_repair

def _apply_sorted_insertion(state: RvrpState, data: ProblemData):
    MAX_ROUTES = 50
    while state.unassigned:
        customer = state.unassigned.pop(0) 
        best_diff = float('inf')
        best_r_idx = -1
        best_pos = -1
        best_metrics = None
        best_vehicle = None
        
        # --- USE HELPER ---
        top_routes = _get_candidate_routes(data, state, customer, limit=MAX_ROUTES)
        
        for _, r_idx, route in top_routes:
            prev_cost = route.cost 
            for i in range(len(route.node_sequence) + 1):
                cost, metrics, veh = calculate_insertion_cost(data, route, customer, i)
                if metrics:
                    diff = cost - prev_cost
                    if diff < best_diff:
                        best_diff = diff
                        best_r_idx = r_idx
                        best_pos = i
                        best_metrics = metrics
                        best_vehicle = veh
        
        new_route = _try_create_new_route(data, customer)
        if new_route:
            if new_route.cost < best_diff:
                best_diff = new_route.cost
                best_r_idx = -1
        
        if best_r_idx != -1:
            target = state.routes[best_r_idx]
            target.node_sequence.insert(best_pos, customer)
            target.total_dist_meters = best_metrics["total_dist_meters"]
            target.total_duration_min = best_metrics["total_duration_min"]
            target.total_wait_time_min = best_metrics["total_wait_time_min"]
            target.total_load_kg = best_metrics["total_load_kg"]
            target.total_load_cbm = best_metrics["total_load_cbm"]
            target.update_centroid(data)
            if target.vehicle_type.type_id != best_vehicle.type_id:
                target.vehicle_type = best_vehicle
        elif new_route:
            state.routes.append(new_route)
            
    return state

def create_largest_demand_repair_operator(data: ProblemData):
    def largest_demand_repair(state: RvrpState, rng, data: ProblemData):
        state.unassigned.sort(key=lambda c: data.demands_kg[c], reverse=True)
        return _apply_sorted_insertion(state, data)
    return largest_demand_repair

def create_earliest_tw_repair_operator(data: ProblemData):
    def earliest_tw_repair(state: RvrpState, rng, data: ProblemData):
        state.unassigned.sort(key=lambda c: data.time_windows[c][0])
        return _apply_sorted_insertion(state, data)
    return earliest_tw_repair

def create_closest_to_depot_repair_operator(data: ProblemData):
    depot_dists = data.dist_matrix[0, :]
    def closest_repair(state: RvrpState, rng, data: ProblemData):
        state.unassigned.sort(key=lambda c: depot_dists[c])
        return _apply_sorted_insertion(state, data)
    return closest_repair
    
def create_farthest_insertion_repair_operator(data: ProblemData):
    depot_dists = data.dist_matrix[0, :]
    def farthest_repair(state: RvrpState, rng, data: ProblemData):
        state.unassigned.sort(key=lambda c: depot_dists[c], reverse=True) # Farthest first
        return _apply_sorted_insertion(state, data)
    return farthest_repair

