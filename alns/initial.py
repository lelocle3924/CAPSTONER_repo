import numpy as np
from core.data_structures import RvrpState, Route, ProblemData, VehicleType
from .repairopt import check_sequence_feasibility 

#ver 5

def _find_best_vehicle_for_sequence(data: ProblemData, sequence: list[int]) -> tuple[VehicleType, dict]:
    """
    Tìm xe rẻ nhất thỏa mãn sequence.
    Trả về (VehicleType, Metrics) hoặc (None, None)
    """
    # Sort fleet by fixed cost (ưu tiên xe nhỏ/rẻ)
    sorted_fleet = sorted(data.vehicle_types, key=lambda v: v.fixed_cost)
    
    for v in sorted_fleet:
        is_feasible, metrics, _ = check_sequence_feasibility(data, v, sequence)
        if is_feasible:
            return v, metrics
            
    return None, None

def clarke_wright_heterogeneous(data: ProblemData) -> RvrpState:
    """
    Parallel Clarke-Wright Savings adapted for Heterogeneous Fleet.
    Fixed: Removed 'set' usage for Route objects to avoid TypeError.
    """
    
    # 1. Init: Mỗi khách 1 route riêng biệt với xe bé nhất (Best Fit)
    customers = [i for i in range(1, data.num_nodes)]
    initial_routes = []
    
    # Map: Node ID -> Route Object mà node đó đang thuộc về
    node_to_route: dict[int, Route] = {}
    
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
            node_to_route[cust] = r
        else:
            # Khách này không xe nào chở được (lẻ loi) -> Sẽ thành Unassigned
            pass 

    # 2. Calculate Savings
    # S_ij = Dist(i,0) + Dist(0,j) - Dist(i,j)
    savings = []
    for i in customers:
        for j in customers:
            if i >= j: continue
            # Chỉ tính saving nếu cả 2 node đều đã được gán route ban đầu
            if i in node_to_route and j in node_to_route:
                s = data.dist_matrix[i, 0] + data.dist_matrix[0, j] - data.dist_matrix[i, j]
                if s > 0:
                    savings.append((s, i, j))
    
    # Sort giảm dần (ưu tiên merge lợi nhất trước)
    savings.sort(key=lambda x: x[0], reverse=True)

    # 3. Merge Loop
    for s_val, i, j in savings:
        # Lấy route hiện tại chứa i và j
        r_i = node_to_route.get(i)
        r_j = node_to_route.get(j)
        
        # Nếu một trong 2 node không còn route (đã bị xử lý lỗi) hoặc
        # Hai node CÙNG nằm trong 1 route rồi -> Skip
        if r_i is None or r_j is None or r_i is r_j:
            continue
            
        # Kiểm tra topo: i phải là điểm cuối r_i, j là điểm đầu r_j HOẶC ngược lại
        # (Standard CW logic)
        i_is_start = (r_i.node_sequence[0] == i)
        i_is_end = (r_i.node_sequence[-1] == i)
        j_is_start = (r_j.node_sequence[0] == j)
        j_is_end = (r_j.node_sequence[-1] == j)
        
        merge_seq = None
        
        # Case 1: ... -> i  +  j -> ...
        if i_is_end and j_is_start:
            merge_seq = r_i.node_sequence + r_j.node_sequence
            
        # Case 2: ... -> j  +  i -> ...
        elif j_is_end and i_is_start:
            merge_seq = r_j.node_sequence + r_i.node_sequence
            
        # Case 3: Đảo ngược route để khớp (Optional, nhưng tăng khả năng merge)
        # Tạm thời bỏ qua để giữ logic đơn giản và đúng hướng luồng (Time Window)
            
        if merge_seq:
            # Tìm xe phù hợp cho chuỗi đã gộp (UPGRADE VEHICLE logic)
            new_vehicle, metrics = _find_best_vehicle_for_sequence(data, merge_seq)
            
            if new_vehicle:
                # MERGE SUCCESS!
                
                # 1. Update r_i thành route mới (Winner takes all)
                r_i.node_sequence = merge_seq
                r_i.vehicle_type = new_vehicle
                r_i.total_dist_meters = metrics['total_dist_meters']
                r_i.total_duration_min = metrics['total_duration_min']
                r_i.total_wait_time_min = metrics['total_wait_time_min']
                r_i.total_load_kg = metrics['total_load_kg']
                r_i.total_load_cbm = metrics['total_load_cbm']
                
                # 2. Update pointers: Tất cả node trong r_j giờ thuộc về r_i
                for node in r_j.node_sequence:
                    node_to_route[node] = r_i
                
                # 3. Mark r_j as empty/dead (để lọc sau)
                r_j.node_sequence = []
                
    # 4. Finalize: Lọc bỏ các route rỗng
    final_routes = [r for r in initial_routes if len(r.node_sequence) > 0]
    
    # Tìm unassigned (những node không có trong map ban đầu)
    unassigned = [i for i in range(1, data.num_nodes) if i not in node_to_route]
    
    return RvrpState(final_routes, unassigned)

# Wrapper function để tương thích ngược
def one_for_one(data):
    return clarke_wright_heterogeneous(data)

def neighbors(dist_matrix, customer_idx):
    locations = np.argsort(dist_matrix[customer_idx])
    return locations[locations != 0]