# file: alns/vrp4ppo/destroyopt.py
# ver 4
# ver 4.1: thêm eliminate small routes
# ver 5: thêm accelerated downgrade logic

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from core.data_structures import RvrpState, ProblemData, Route, VehicleType
from .initial import neighbors, check_sequence_feasibility


# ============================================================
# 1. CONFIGURATION
# ============================================================

def get_destroy_params(num_nodes):
    """
    Cấu hình mặc định cho các destroy operators
    """
    n_customers = num_nodes - 1
    base_destroy_ratio = 0.15 if n_customers <= 50 else 0.1
    
    return {
        "random_removal": {
            "min": max(1, int(n_customers * 0.05)),
            "max": max(2, int(n_customers * base_destroy_ratio))
        },
        "worst_removal": {
            "min": max(1, int(n_customers * 0.05)),
            "max": max(2, int(n_customers * base_destroy_ratio)),
            "alpha": 0.6 
        },
        "string_removal": {
            "max_string_size": 5, 
        },
        # Operator mới: Low Utilization Route Removal
        "low_util_route_removal": {
        }
    }

# ============================================================
# 2. LOGIC UPDATE STATE & DOWNGRADE VEHICLES IF POSSIBLE
# ============================================================

def _find_cheapest_feasible_vehicle(data: ProblemData, current_route: Route) -> Route:
    """
    [OPTIMIZED] Finds the cheapest vehicle for the remaining sequence.
    Uses the Numba-backed engine to check feasibility at C-speed.
    """
    if not current_route.node_sequence:
        return current_route

    # Sort fleet by fixed cost (usually smallest capacity first)
    sorted_fleet = sorted(data.vehicle_types, key=lambda v: v.fixed_cost)
    
    current_cost = current_route.cost
    
    for v in sorted_fleet:
        # Optimization: if fixed cost is already higher than current, stop (since list is sorted)
        if v.fixed_cost >= current_route.vehicle_type.fixed_cost:
            break
            
        # Use the high-speed Numba engine
        is_feas, metrics, _ = check_sequence_feasibility(data, v, current_route.node_sequence)
        
        if is_feas:
            # We found a cheaper vehicle that can handle the reduced load/time
            current_route.vehicle_type = v
            current_route.total_dist_meters = metrics['total_dist_meters']
            current_route.total_duration_min = metrics['total_duration_min']
            current_route.total_wait_time_min = metrics['total_wait_time_min']
            current_route.total_load_kg = metrics['total_load_kg']
            current_route.total_load_cbm = metrics['total_load_cbm']
            return current_route
                
    return current_route

# def _find_cheapest_feasible_vehicle(data: ProblemData, current_route: Route) -> Route:
#     mỗi lần check là lại loop qua toàn bộ fleet, quá dài
# if not current_route.node_sequence: return current_route

#     # Sort fleet vẫn giữ nguyên để ưu tiên check xe nhỏ trước
#     sorted_fleet = sorted(data.vehicle_types, key=lambda v: v.fixed_cost)
    
#     best_vehicle = current_route.vehicle_type
#     min_total_cost = current_route.cost # Cost hiện tại là benchmark
#     found_better = False
#     best_metrics = None
    
#     for v in sorted_fleet:
#         # [TUAN FIX]: Bỏ đoạn break if fixed_cost > current.fixed_cost
#         # Chỉ skip nếu xe yếu hơn tải trọng cần thiết (để đỡ tốn công tính Numba)
#         # (Đoạn này phải tính total load trước vòng lặp)
        
#         # Gọi Numba
#         is_feas, metrics, _ = check_sequence_feasibility(data, v, current_route.node_sequence)
        
#         if is_feas:
#             # Tính Total Cost chuẩn
#             dist_km = metrics['total_dist_meters'] / 1000.0
#             dur_hr = metrics['total_duration_min'] / 60.0
#             new_cost = v.fixed_cost + dist_km * v.cost_per_km + dur_hr * v.cost_per_hour
            
#             # So sánh sòng phẳng
#             if new_cost < min_total_cost:
#                 min_total_cost = new_cost
#                 best_vehicle = v
#                 best_metrics = metrics
#                 found_better = True
                
#     if found_better:
#         current_route.vehicle_type = best_vehicle
#         current_route.total_dist_meters = best_metrics['total_dist_meters']
#         current_route.total_duration_min = best_metrics['total_duration_min']
#         current_route.total_wait_time_min = best_metrics['total_wait_time_min']
#         current_route.total_load_kg = best_metrics['total_load_kg']
#         current_route.total_load_cbm = best_metrics['total_load_cbm']
        
#     return current_route

def update_single_route_metrics(route: Route, data: ProblemData):
    """Update metrics and attempt vehicle downgrade."""
    if not route.node_sequence:
        route.total_dist_meters = 0
        return

    # Update metrics with current vehicle first
    is_feas, metrics, _ = check_sequence_feasibility(data, route.vehicle_type, route.node_sequence)
    if metrics:
        route.total_dist_meters = metrics['total_dist_meters']
        route.total_duration_min = metrics['total_duration_min']
        route.total_wait_time_min = metrics['total_wait_time_min']
        route.total_load_kg = metrics['total_load_kg']
        route.total_load_cbm = metrics['total_load_cbm']
    
    # Attempt Downgrade to save cost
    _find_cheapest_feasible_vehicle(data, route)
    # Update spatial centroid for the next repair step
    route.update_centroid(data)

def update_destroyed_state(state: RvrpState, data: ProblemData) -> RvrpState:
    """Cleanup empty routes and refresh metrics."""
    state.routes = [r for r in state.routes if len(r.node_sequence) > 0]
    for route in state.routes:
        update_single_route_metrics(route, data)
    return state
# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def _find_route_containing_node(state: RvrpState, node: int) -> Route:
    for route in state.routes:
        if node in route.node_sequence:
            return route
    return None

def _calc_removal_gain(route: Route, node: int, data: ProblemData) -> float:
    # Gain xấp xỉ bằng Distance reduction (chưa tính Time reduction để nhanh)
    seq = route.node_sequence
    idx = seq.index(node)
    prev_node = 0 if idx == 0 else seq[idx - 1]
    next_node = 0 if idx == len(seq) - 1 else seq[idx + 1]
    dist = data.dist_matrix
    
    # Cost cũ - Cost mới
    # Gain dương nghĩa là giảm được khoảng cách
    return dist[prev_node, node] + dist[node, next_node] - dist[prev_node, next_node]

# ============================================================
# 4. DESTROY OPERATORS
# ============================================================
def create_random_customer_removal_operator(data: ProblemData, ratio: float = 0.1):
    def random_removal(state: RvrpState, rng, data: ProblemData):
        destroyed = state.copy()
        assigned_nodes = [n for r in destroyed.routes for n in r.node_sequence]
        if not assigned_nodes: return destroyed

        num_remove = max(1, int(len(assigned_nodes) * ratio))
        targets = rng.choice(assigned_nodes, num_remove, replace=False)

        # Optimization: use a set for faster lookup
        target_set = set(targets)
        for route in destroyed.routes:
            route.node_sequence = [n for n in route.node_sequence if n not in target_set]
        
        destroyed.unassigned.extend(targets)
        return update_destroyed_state(destroyed, data)
    return random_removal


def create_worst_removal_operator(data: ProblemData, ratio: float = None):
    defaults = get_destroy_params(data.num_nodes)["worst_removal"]

    def worst_removal(state: RvrpState, rng, data: ProblemData):
        destroyed = state.copy()
        n_cust = data.num_nodes - 1
        
        if ratio is not None:
            num_remove = int(n_cust * ratio)
        else:
            num_remove = rng.integers(defaults["min"], defaults["max"] + 1)
        
        num_remove = max(1, min(num_remove, n_cust))

        for _ in range(num_remove):
            costs = []
            for route in destroyed.routes:
                for node in route.node_sequence:
                    gain = _calc_removal_gain(route, node, data)
                    costs.append((gain, node, route))
            
            if not costs: break
            
            costs.sort(key=lambda x: x[0], reverse=True)
            
            p = defaults["alpha"]
            idx = int(len(costs) * (rng.random()**p))
            idx = min(idx, len(costs)-1)
            
            _, target_node, target_route = costs[idx]
            
            target_route.node_sequence.remove(target_node)
            destroyed.unassigned.append(target_node)
            
        return update_destroyed_state(destroyed, data) # UPDATE HERE
    return worst_removal


def create_random_route_removal_operator(data: ProblemData, ratio: float = None):
    """
    Xóa ngẫu nhiên các route
    """
    def route_removal(state: RvrpState, rng, data: ProblemData):
        destroyed = state.copy()
        if not destroyed.routes: return destroyed
        
        num_routes = len(destroyed.routes)
        
        if ratio is not None:
            count = int(num_routes * ratio)
        else:
            count = 1
            
        count = max(1, min(count, num_routes))
        
        indices_to_remove = rng.choice(num_routes, count, replace=False)
        indices_to_remove.sort() # Sort để pop từ dưới lên
        
        for idx in reversed(indices_to_remove):
            removed_route = destroyed.routes.pop(idx)
            destroyed.unassigned.extend(removed_route.node_sequence)
        
        return update_destroyed_state(destroyed, data) # Data is unused here but kept for interface
    return route_removal


def create_low_utilization_route_removal_operator(data: ProblemData, ratio: float = None):
    def low_util_route_removal(state: RvrpState, rng, data: ProblemData):
        destroyed = state.copy()
        if not destroyed.routes: return destroyed
        
        num_routes = len(destroyed.routes)
        
        # 1. Xác định số lượng route cần xóa
        if ratio is not None:
            count = int(num_routes * ratio)
        else:
            count = 1
        
        count = max(1, min(count, num_routes))
        
        # 2. Sắp xếp routes theo utilization từ Thấp -> Cao
        # Lưu index cũ để xóa
        route_utils = []
        for i, r in enumerate(destroyed.routes):
            route_utils.append((r.capacity_utilization, i))
            
        route_utils.sort(key=lambda x: x[0]) # Ascending sort
        
        # 3. Lấy n routes thấp nhất
        indices_to_remove = [x[1] for x in route_utils[:count]]
        indices_to_remove.sort() # Sort index để pop an toàn
        
        # 4. Xóa và đẩy khách ra unassigned
        for idx in reversed(indices_to_remove):
            removed_route = destroyed.routes.pop(idx)
            destroyed.unassigned.extend(removed_route.node_sequence)
            
        # 5. Update state
        # Mặc dù xóa route không ảnh hưởng metrics route khác, nhưng gọi để đảm bảo consistency
        return update_destroyed_state(destroyed, data)
        
    return low_util_route_removal


def create_string_removal_operator(data: ProblemData, ratio: float = None):
    defaults = get_destroy_params(data.num_nodes)["string_removal"]
    MAX_STRING_SIZE = defaults.get("max_string_size", 5)

    def string_removal(state: RvrpState, rng, data: ProblemData):
        destroyed = state.copy()
        n_cust = data.num_nodes - 1
        
        if ratio is not None:
            target_remove = int(n_cust * ratio)
        else:
            target_remove = rng.integers(2, max(5, int(n_cust * 0.15)))
        target_remove = max(1, min(target_remove, n_cust))
        
        # Chọn tâm
        center_node = rng.integers(1, data.num_nodes)
        neighbor_nodes = neighbors(data.dist_matrix, center_node)
        
        removed_count = 0
        
        for customer in neighbor_nodes:
            if removed_count >= target_remove: break
            if customer in destroyed.unassigned: continue
                
            route = _find_route_containing_node(destroyed, customer)
            if not route: continue
            
            seq = route.node_sequence
            cust_idx = seq.index(customer)
            
            remaining_quota = target_remove - removed_count
            string_len = rng.integers(1, min(remaining_quota, MAX_STRING_SIZE) + 1)
            
            start = max(0, cust_idx - string_len // 2)
            end = min(len(seq), start + string_len)
            
            nodes_to_remove = seq[start:end]
            
            # Remove safely
            route.node_sequence = [x for x in seq if x not in nodes_to_remove]
            
            destroyed.unassigned.extend(nodes_to_remove)
            removed_count += len(nodes_to_remove)
            
        return update_destroyed_state(destroyed, data) # UPDATE HERE

    return string_removal

def create_related_removal_operator(state: RvrpState, ratio: float = None):
    # (Shaw Removal)
    def calculate_relatedness(i, j, dist_matrix, data: ProblemData):
        dist = dist_matrix[i, j]
        # Có thể thêm logic same_route nếu cần, nhưng đơn giản hóa dùng khoảng cách
        return 1.0 / (dist + 0.01)

    def related_removal(state: RvrpState, rng, data: ProblemData):
        destroyed = state.copy()
        n_cust = data.num_nodes - 1
        
        if ratio is not None:
            target_count = int(n_cust * ratio)
        else:
            target_count = rng.integers(2, max(3, int(n_cust * 0.15)))
            
        target_count = max(1, min(target_count, n_cust))

        # 1. Chọn 1 seed customer ngẫu nhiên để xóa
        assigned_nodes = [n for r in destroyed.routes for n in r.node_sequence]
        if not assigned_nodes: return destroyed
        
        seed_node = rng.choice(assigned_nodes)
        
        # Xóa seed node
        route = _find_route_containing_node(destroyed, seed_node)
        if route:
            route.node_sequence.remove(seed_node)
            destroyed.unassigned.append(seed_node)
            
        # 2. Xóa các node "liên quan" nhất đến seed node
        while len(destroyed.unassigned) < target_count:
            current_assigned = [n for r in destroyed.routes for n in r.node_sequence]
            if not current_assigned: break
            
            # Chọn ngẫu nhiên 1 node đã bị xóa để làm mốc so sánh
            ref_node = rng.choice(destroyed.unassigned)
            
            # Tính relatedness với tất cả node còn lại
            # (Tạm dùng Euclidean/Distance matrix làm thước đo)
            rels = []
            for cand in current_assigned:
                rel = calculate_relatedness(ref_node, cand, data.dist_matrix, data)
                rels.append((rel, cand))
            
            rels.sort(key=lambda x: x[0], reverse=True) # Liên quan nhất xếp đầu
            
            # Chọn ngẫu nhiên trong top (Shaw selection)
            idx = int(len(rels) * (rng.random()**4)) # Power 4
            idx = min(idx, len(rels)-1)
            
            to_remove = rels[idx][1]
            
            route = _find_route_containing_node(destroyed, to_remove)
            if route:
                route.node_sequence.remove(to_remove)
                destroyed.unassigned.append(to_remove)

        return update_destroyed_state(destroyed, data)
    return related_removal

def create_sequence_removal_operator(data: ProblemData, ratio: float = None):
    """
    Sequence Removal:
    Remove contiguous segments in a random route.
    """
    def sequence_removal(state: RvrpState, rng, data: ProblemData):
        destroyed = state.copy()
        n_cust = data.num_nodes - 1
        
        if ratio is not None:
            target_remove = int(n_cust * ratio)
        else:
            target_remove = rng.integers(2, max(5, int(n_cust * 0.15)))
            
        target_remove = max(1, min(target_remove, n_cust))
        removed_count = 0
        
        while removed_count < target_remove:
            valid_routes = [r for r in destroyed.routes if len(r.node_sequence) > 0]
            if not valid_routes: break
            
            route = rng.choice(valid_routes)
            seq = route.node_sequence
            
            max_cut = max(1, len(seq) // 3)
            remaining = target_remove - removed_count
            cut_len = rng.integers(1, min(remaining, max_cut) + 1)
            
            start_idx = rng.integers(0, len(seq) - cut_len + 1)
            end_idx = start_idx + cut_len
            
            nodes_to_remove = seq[start_idx:end_idx]
            
            del route.node_sequence[start_idx:end_idx]
            
            destroyed.unassigned.extend(nodes_to_remove)
            removed_count += len(nodes_to_remove)

        return update_destroyed_state(destroyed, data)

    return sequence_removal

def create_eliminate_small_route_operator(data: ProblemData, min_stops: int = 2):
    """
    Operator chuyên dụng để dọn dẹp các route 'vụn vặt'.
    Nếu một route chỉ phục vụ <= min_stops khách hàng, nó sẽ bị hủy.
    Khách hàng trong route đó bị đẩy ra unassigned để các route lớn hơn 'gom' lấy.
    """
    def eliminate_small_routes(state: RvrpState, rng, data: ProblemData):
        destroyed = state.copy()
        
        # Nếu không còn route nào thì return
        if not destroyed.routes:
            return destroyed

        # Tìm các route quá ngắn
        # Duyệt ngược để delete an toàn
        indices_to_remove = []
        for i in range(len(destroyed.routes) - 1, -1, -1):
            route = destroyed.routes[i]
            # Route.node_sequence chứa các khách hàng (không tính depot)
            if len(route.node_sequence) <= min_stops:
                indices_to_remove.append(i)
        
        # Thực hiện xóa
        for idx in indices_to_remove:
            removed_route = destroyed.routes.pop(idx)
            destroyed.unassigned.extend(removed_route.node_sequence)
            
        # Update lại state (metrics, pointers...)
        return update_destroyed_state(destroyed, data)

    return eliminate_small_routes