# file: alns/vrp4ppo/repairopt.py

import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any
from core.data_structures import RvrpState, Route, ProblemData, VehicleType

# ============================================================
# 1. FEASIBILITY ENGINE & COST CALCULATOR (PURE FUNCTIONS)
# ============================================================

def check_sequence_feasibility(
    data: ProblemData, 
    vehicle: VehicleType, 
    node_sequence: List[int]
) -> Tuple[bool, Optional[Dict[str, float]], str]:
    """
    Kiểm tra tính khả thi của chuỗi node với loại xe cụ thể.
    Check 5 constraints: Preference, KG, CBM, Time Window, Max Duration.
    """
    
    # 1. PREFERENCE CHECK (O(N)) - Fail fast
    for node in node_sequence:
        if vehicle.type_id not in data.allowed_vehicles[node]:
            return False, None, f"Preference Violation at Node {node}"

    # 2. CAPACITY CHECK (O(N)) - Fail fast
    # Lưu ý: node_sequence ở đây KHÔNG chứa Depot (theo convention của one_for_one/RvrpState)
    total_kg = 0.0
    total_cbm = 0.0
    for node in node_sequence:
        total_kg += data.demands_kg[node]
        total_cbm += data.demands_cbm[node]
        
    if total_kg > vehicle.capacity_kg:
        return False, None, f"Capacity KG: {total_kg} > {vehicle.capacity_kg}"
    if total_cbm > vehicle.capacity_cbm:
        return False, None, f"Capacity CBM: {total_cbm} > {vehicle.capacity_cbm}"

    # 3. TIME SIMULATION (O(N))
    current_time = data.time_windows[0][0] # Depot Open
    total_dist = 0.0
    total_wait = 0.0
    
    prev_node = 0 # Start at Depot
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
            return False, None, f"TW Violation Node {node}: Arr {current_time} > {end_window}"
        
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
    
    # Check Depot Closing
    if current_time > data.time_windows[0][1]:
        return False, None, "Depot Closing Violation"
        
    # Check Max Duration
    total_duration = current_time - data.time_windows[0][0]
    if total_duration > data.max_route_duration:
        return False, None, f"Max Duration Violation: {total_duration} > {data.max_route_duration}"

    # SUCCESS
    metrics = {
        "total_dist_meters": total_dist,
        "total_duration_min": total_duration,
        "total_wait_time_min": total_wait,
        "total_load_kg": total_kg,
        "total_load_cbm": total_cbm
    }
    return True, metrics, "OK"


def calculate_insertion_cost(
    data: ProblemData,
    route: Route,
    customer: int,
    idx: int
) -> Tuple[float, Optional[Dict[str, float]]]:
    """
    Tính chi phí TUYỆT ĐỐI nếu chèn customer vào vị trí idx.
    Returns: (New_Absolute_Cost, Metrics). Return (inf, None) nếu infeasible.
    """
    # Create temp sequence
    new_sequence = route.node_sequence[:]
    new_sequence.insert(idx, customer)
    
    # Check Feasibility
    is_feasible, metrics, _ = check_sequence_feasibility(data, route.vehicle_type, new_sequence)
    
    if not is_feasible:
        return float('inf'), None
        
    # Calculate Economic Cost
    v = route.vehicle_type
    dist_km = metrics["total_dist_meters"] / 1000.0
    dur_hr = metrics["total_duration_min"] / 60.0
    
    # Cost = Fixed + Variable
    total_cost = v.fixed_cost + (dist_km * v.cost_per_km) + (dur_hr * v.cost_per_hour)
    
    return total_cost, metrics

# ============================================================
# 2. HELPER: NEW ROUTE CREATION STRATEGY
# ============================================================

def _try_create_new_route(data: ProblemData, customer: int) -> Optional[Route]:
    """
    Tìm loại xe BÉ NHẤT có thể phục vụ khách hàng này (hoặc chuỗi này).
    Trả về Route mới hoặc None nếu không xe nào phục vụ được.
    """
    # Sort fleet by fixed cost (usually implies size)
    sorted_fleet = sorted(data.vehicle_types, key=lambda x: x.fixed_cost)
    
    for v in sorted_fleet:
        is_feasible, metrics, _ = check_sequence_feasibility(data, v, [customer])
        
        if is_feasible:
            # Create fresh route
            return Route(
                vehicle_type=v,
                node_sequence=[customer],
                total_dist_meters=metrics["total_dist_meters"],
                total_duration_min=metrics["total_duration_min"],
                total_wait_time_min=metrics["total_wait_time_min"],
                total_load_kg=metrics["total_load_kg"],
                total_load_cbm=metrics["total_load_cbm"],
                is_time_feasible=True,
                is_capacity_feasible=True,
                is_duration_feasible=True,
                is_preference_feasible=True
            )
            
    return None

# ============================================================
# 3. REPAIR OPERATORS
# ============================================================

def create_greedy_repair_operator(data: ProblemData):
    
    def greedy_repair(state: RvrpState, rng, data: ProblemData):
        """
        Greedy Insertion:
        1. Shuffle unassigned.
        2. Với mỗi khách, tìm vị trí chèn rẻ nhất trong TẤT CẢ route hiện có.
        3. Nếu không chèn được vào đâu -> Tạo route mới.
        """
        rng.shuffle(state.unassigned)
        
        while state.unassigned:
            customer = state.unassigned.pop()
            
            best_cost_increase = float('inf')
            best_route_idx = -1
            best_insert_pos = -1
            best_metrics = None
            
            # --- PHASE 1: Try Inserting into Existing Routes ---
            for r_idx, route in enumerate(state.routes):
                
                # Pruning nhanh (Capacity & Preference)
                if route.vehicle_type.type_id not in data.allowed_vehicles[customer]:
                    continue
                # Ước lượng capacity (chưa chính xác 100% nhưng loại bỏ nhanh)
                if (route.total_load_kg + data.demands_kg[customer] > route.vehicle_type.capacity_kg):
                    continue
                
                prev_cost = route.cost 
                
                # Duyệt mọi vị trí
                for i in range(len(route.node_sequence) + 1):
                    new_abs_cost, metrics = calculate_insertion_cost(data, route, customer, i)
                    
                    if metrics is not None:
                        # Marginal Cost = New Absolute Cost - Old Absolute Cost
                        cost_increase = new_abs_cost - prev_cost
                        
                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_route_idx = r_idx
                            best_insert_pos = i
                            best_metrics = metrics
            
            # --- PHASE 2: Apply Move or Create New ---
            if best_route_idx != -1:
                # INSERT VÀO ROUTE CŨ
                target_route = state.routes[best_route_idx]
                target_route.node_sequence.insert(best_insert_pos, customer)
                
                # UPDATE METRICS IMMEDIATELY (No stale state)
                target_route.total_dist_meters = best_metrics["total_dist_meters"]
                target_route.total_duration_min = best_metrics["total_duration_min"]
                target_route.total_wait_time_min = best_metrics["total_wait_time_min"]
                target_route.total_load_kg = best_metrics["total_load_kg"]
                target_route.total_load_cbm = best_metrics["total_load_cbm"]
                
            else:
                # TẠO ROUTE MỚI
                new_route = _try_create_new_route(data, customer)
                if new_route:
                    state.routes.append(new_route)
                    # Route mới này sẽ được các vòng lặp sau coi là "Existing Route" 
                    # để nhét thêm khách khác vào -> Đây chính là logic gom đơn
                else:
                    # Bó tay toàn tập (do ràng buộc quá chặt)
                    # Push back to solve later (hoặc chấp nhận mất khách)
                    # Ở đây ta tạm thời append lại vào cuối để xử lý (nhưng cẩn thận infinite loop)
                    # Tốt nhất là bỏ qua và chấp nhận Unassigned penalty trong Objective
                    pass 
                    
        return state

    return greedy_repair


def create_criticality_repair_operator(data: ProblemData):
    """
    Giống Greedy, nhưng sort unassigned theo độ khó (Criticality) trước.
    Khách khó (xa, demand to, time window hẹp) làm trước.
    """
    # Pre-calc importance values
    min_tw = 30.0 # avoid div by 0
    max_tw = data.time_windows[0][1] - data.time_windows[0][0]
    
    # Cache depot distances
    depot_dists = data.dist_matrix[0, :]
    
    def get_importance(node_idx):
        # 1. Demand score
        norm_dem = data.demands_kg[node_idx] / 1000.0 # simple scaling
        # 2. Distance score
        norm_dist = depot_dists[node_idx] / 10000.0
        # 3. TW score (càng hẹp càng quan trọng)
        tw_len = data.time_windows[node_idx][1] - data.time_windows[node_idx][0]
        norm_tw = 1.0 / (max(tw_len, min_tw) / 60.0)
        
        return norm_dem + norm_dist + norm_tw

    def criticality_repair(state: RvrpState, rng, data: ProblemData):
        state.unassigned.sort(key=get_importance) # Pop from end -> Sort Ascending is acceptable if using pop(), but list.pop(0) is slow.

        while state.unassigned:
            customer = state.unassigned.pop() # Lấy ông quan trọng nhất
            
            best_cost_increase = float('inf')
            best_route_idx = -1
            best_insert_pos = -1
            best_metrics = None
            
            # --- Try Insert ---
            for r_idx, route in enumerate(state.routes):
                if route.vehicle_type.type_id not in data.allowed_vehicles[customer]: continue
                if (route.total_load_kg + data.demands_kg[customer] > route.vehicle_type.capacity_kg): continue
                
                prev_cost = route.cost 
                
                for i in range(len(route.node_sequence) + 1):
                    new_abs_cost, metrics = calculate_insertion_cost(data, route, customer, i)
                    if metrics:
                        cost_increase = new_abs_cost - prev_cost
                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_route_idx = r_idx
                            best_insert_pos = i
                            best_metrics = metrics
                            
            # --- Apply ---
            if best_route_idx != -1:
                target_route = state.routes[best_route_idx]
                target_route.node_sequence.insert(best_insert_pos, customer)
                target_route.total_dist_meters = best_metrics["total_dist_meters"]
                target_route.total_duration_min = best_metrics["total_duration_min"]
                target_route.total_wait_time_min = best_metrics["total_wait_time_min"]
                target_route.total_load_kg = best_metrics["total_load_kg"]
                target_route.total_load_cbm = best_metrics["total_load_cbm"]
            else:
                new_route = _try_create_new_route(data, customer)
                if new_route: state.routes.append(new_route)

        return state
        
    return criticality_repair


def create_regret_repair_operator(data: ProblemData, k: int = 2):
    """
    Regret-2 Repair:
    Với mỗi khách unassigned, tính chênh lệch chi phí giữa Best Route và 2nd Best Route.
    Khách nào có chênh lệch lớn nhất (Regret Max) sẽ được chèn trước.
    """
    def regret_repair(state: RvrpState, rng, data: ProblemData):
        
        while state.unassigned:
            # Lưu thông tin regret cho tất cả candidate
            # Format: (regret_value, customer_id, best_r_idx, best_pos, best_metrics)
            candidates_regret = []
            
            for customer in state.unassigned:
                # Tìm top 2 insertions cho khách này
                valid_insertions = [] # list of (cost_increase, r_idx, pos, metrics)
                
                # 1. Quét qua các route hiện tại
                for r_idx, route in enumerate(state.routes):
                    if route.vehicle_type.type_id not in data.allowed_vehicles[customer]: continue
                    if (route.total_load_kg + data.demands_kg[customer] > route.vehicle_type.capacity_kg): continue
                    
                    prev_cost = route.cost
                    
                    # Tìm best pos trong route này
                    local_best_inc = float('inf')
                    local_best_pos = -1
                    local_metrics = None
                    
                    for i in range(len(route.node_sequence) + 1):
                        new_cost, metrics = calculate_insertion_cost(data, route, customer, i)
                        if metrics:
                            inc = new_cost - prev_cost
                            if inc < local_best_inc:
                                local_best_inc = inc
                                local_best_pos = i
                                local_metrics = metrics
                    
                    if local_best_pos != -1:
                        valid_insertions.append((local_best_inc, r_idx, local_best_pos, local_metrics))
                
                # 2. Quét qua khả năng tạo New Route (coi như 1 option)
                # Tính cost tạo new route
                dummy_new = _try_create_new_route(data, customer)
                if dummy_new:
                    # Cost increase = Full cost of new route (vì prev cost = 0)
                    valid_insertions.append((dummy_new.cost, -1, 0, {
                        "total_dist_meters": dummy_new.total_dist_meters,
                        "total_duration_min": dummy_new.total_duration_min,
                        "total_wait_time_min": dummy_new.total_wait_time_min,
                        "total_load_kg": dummy_new.total_load_kg,
                        "total_load_cbm": dummy_new.total_load_cbm
                    }))
                
                # 3. Tính Regret
                if not valid_insertions:
                    continue # Khách này vô vọng
                
                # Sort insertions by cost increase ascending
                valid_insertions.sort(key=lambda x: x[0])
                
                best = valid_insertions[0]
                
                if len(valid_insertions) >= k:
                    second = valid_insertions[k-1] # Regret-k uses k-th best
                    regret_val = second[0] - best[0]
                else:
                    # Nếu chỉ có 1 lựa chọn -> Regret cực lớn (phải chèn ngay kẻo mất)
                    regret_val = float('inf')
                    
                candidates_regret.append({
                    "regret": regret_val,
                    "customer": customer,
                    "r_idx": best[1],
                    "pos": best[2],
                    "metrics": best[3]
                })
            
            if not candidates_regret:
                break # Không còn khách nào chèn được nữa
            
            # Chọn khách có Regret lớn nhất
            # Sort desc by regret
            candidates_regret.sort(key=lambda x: x["regret"], reverse=True)
            winner = candidates_regret[0]
            
            # Execute Insertion
            cust_to_insert = winner["customer"]
            state.unassigned.remove(cust_to_insert)
            
            if winner["r_idx"] != -1:
                # Insert vào existing
                target_route = state.routes[winner["r_idx"]]
                target_route.node_sequence.insert(winner["pos"], cust_to_insert)
                m = winner["metrics"]
                target_route.total_dist_meters = m["total_dist_meters"]
                target_route.total_duration_min = m["total_duration_min"]
                target_route.total_wait_time_min = m["total_wait_time_min"]
                target_route.total_load_kg = m["total_load_kg"]
                target_route.total_load_cbm = m["total_load_cbm"]
            else:
                # Create new
                new_route = _try_create_new_route(data, cust_to_insert)
                state.routes.append(new_route)

        return state

    return regret_repair