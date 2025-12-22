# file: alns/vrp4ppo/initial.py
# ver 4, chỉ giữ lại mỗi clarke-wright thôi
# lỗi: set(initial_routes) không được

import numpy as np
from core.data_structures import RvrpState, Route, ProblemData, VehicleType
from .repairopt import check_sequence_feasibility # Reuse logic check

def _find_best_vehicle_for_sequence(data: ProblemData, sequence: list[int]) -> tuple[VehicleType, dict]:
    """Tìm xe rẻ nhất thỏa mãn sequence"""
    # Sort fleet by fixed cost ~ size
    sorted_fleet = sorted(data.vehicle_types, key=lambda v: v.fixed_cost)
    
    for v in sorted_fleet:
        is_feasible, metrics, _ = check_sequence_feasibility(data, v, sequence)
        if is_feasible:
            return v, metrics
            
    return None, None

def clarke_wright_heterogeneous(data: ProblemData) -> RvrpState:
    """
    Parallel Clarke-Wright Savings adapted for Heterogeneous Fleet.
    """
    routes = []
    
    # 1. Init: Mỗi khách 1 route với xe bé nhất (Best Fit)
    # Re-use logic one_for_one nhưng trả về List[Route]
    customers = [i for i in range(1, data.num_nodes)]
    initial_routes = []
    
    for cust in customers:
        v, metrics = _find_best_vehicle_for_sequence(data, [cust])
        if v:
            r = Route(v, [cust], 
                      total_dist_meters=metrics['total_dist_meters'],
                      total_duration_min=metrics['total_duration_min'],
                      total_wait_time_min=metrics['total_wait_time_min'],
                      total_load_kg=metrics['total_load_kg'],
                      total_load_cbm=metrics['total_load_cbm'])
            initial_routes.append(r)
        else:
            # Unassigned ngay từ đầu
            pass 

    # Map: Node -> Route Object
    node_to_route = {}
    for r in initial_routes:
        node_to_route[r.node_sequence[0]] = r
        
    active_routes = set(initial_routes)

    # 2. Calculate Savings
    # S_ij = Dist(i,0) + Dist(0,j) - Dist(i,j)
    # Tuy cost phụ thuộc loại xe, nhưng Dist Savings vẫn là heuristic tốt để ưu tiên merge các node gần nhau.
    savings = []
    for i in customers:
        for j in customers:
            if i >= j: continue
            s = data.dist_matrix[i, 0] + data.dist_matrix[0, j] - data.dist_matrix[i, j]
            if s > 0:
                savings.append((s, i, j))
    
    savings.sort(key=lambda x: x[0], reverse=True)

    # 3. Merge Loop
    for s_val, i, j in savings:
        r_i = node_to_route.get(i)
        r_j = node_to_route.get(j)
        
        if r_i is None or r_j is None: continue
        if r_i is r_j: continue # Already same route
        
        # Check merge topology (i phải ở cuối r_i, j phải ở đầu r_j hoặc ngược lại)
        # Để đơn giản, CW chuẩn yêu cầu: End(r_i) == i AND Start(r_j) == j
        
        merge_seq = None
        if r_i.node_sequence[-1] == i and r_j.node_sequence[0] == j:
            merge_seq = r_i.node_sequence + r_j.node_sequence
        elif r_j.node_sequence[-1] == j and r_i.node_sequence[0] == i:
            merge_seq = r_j.node_sequence + r_i.node_sequence
            
        if merge_seq:
            # Check Feasibility & Find Vehicle for MERGED sequence
            # Đây là điểm khác biệt: Route gộp có thể cần xe TO HƠN
            new_vehicle, metrics = _find_best_vehicle_for_sequence(data, merge_seq)
            
            if new_vehicle:
                # Merge thành công!
                # Update r_i thành route gộp
                r_i.node_sequence = merge_seq
                r_i.vehicle_type = new_vehicle # Upgrade vehicle
                r_i.total_dist_meters = metrics['total_dist_meters']
                r_i.total_duration_min = metrics['total_duration_min']
                r_i.total_wait_time_min = metrics['total_wait_time_min']
                r_i.total_load_kg = metrics['total_load_kg']
                r_i.total_load_cbm = metrics['total_load_cbm']
                
                # Update map & delete r_j
                for node in r_j.node_sequence:
                    node_to_route[node] = r_i
                
                active_routes.remove(r_j)
                
    # 4. Finalize
    unassigned = [i for i in range(1, data.num_nodes) if i not in node_to_route]
    
    return RvrpState(list(active_routes), unassigned)

def neighbors(dist_matrix, customer_idx):
    locations = np.argsort(dist_matrix[customer_idx])
    return locations[locations != 0]