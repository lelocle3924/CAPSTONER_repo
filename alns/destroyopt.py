# file: alns/vrp4ppo/destroyopt.py
# ver 4
# ver 4.1: thêm eliminate small routes

import numpy as np
from typing import List
from core.data_structures import RvrpState, ProblemData, Route
from .initial import neighbors # Giả định file initial.py nằm cùng thư mục

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
    Helper function: Tìm loại xe rẻ nhất có thể phục vụ route hiện tại.
    Dùng để Downgrade xe sau khi xóa bớt khách.
    """
    # 1. Tính tổng tải trọng hiện tại
    total_kg = sum(data.demands_kg[n] for n in current_route.node_sequence)
    total_cbm = sum(data.demands_cbm[n] for n in current_route.node_sequence)
    
    # 2. Lấy danh sách xe, sort theo Fixed Cost (hoặc Capacity)
    sorted_fleet = sorted(data.vehicle_types, key=lambda v: v.fixed_cost)
    
    best_vehicle = current_route.vehicle_type
    found_better = False
    
    # Chỉ cần check các xe có capacity >= load hiện tại
    # Và rẻ hơn xe hiện tại (Logic tối ưu)
    
    current_cost = current_route.cost # Cost với xe hiện tại
    
    for v in sorted_fleet:
        # Skip nếu xe này quá yếu
        if v.capacity_kg < total_kg or v.capacity_cbm < total_cbm:
            continue
            
        # Skip preference check kỹ ở đây để nhanh (giả định xe nhỏ thường luồn lách tốt hơn)
        # Nhưng để an toàn:
        pref_fail = False
        for node in current_route.node_sequence:
            if v.type_id not in data.allowed_vehicles[node]:
                pref_fail = True
                break
        if pref_fail: continue

        # Nếu xe này là xe hiện tại -> Keep it (trừ khi tìm đc xe khác rẻ hơn ở vòng lặp trước/sau)
        if v.type_id == current_route.vehicle_type.type_id:
            break # Đã đến xe hiện tại, các xe sau đắt hơn (do sorted) -> Stop
            
        # Check Feasibility (Time/Duration)
        # Hàm này cần import từ repairopt hoặc viết lại logic simulation nhỏ ở đây
        # Để tránh circular import, tôi viết logic simulation nhanh ở dưới
        is_feas, metrics = _quick_sim(data, v, current_route.node_sequence)
        
        if is_feas:
            # Tính cost thử
            dist_km = metrics['dist'] / 1000.0
            dur_hr = metrics['duration'] / 60.0
            new_cost = v.fixed_cost + dist_km * v.cost_per_km + dur_hr * v.cost_per_hour
            
            if new_cost < current_cost:
                # Tìm thấy xe rẻ hơn!
                current_route.vehicle_type = v
                # Update metrics theo xe mới
                current_route.total_dist_meters = metrics['dist']
                current_route.total_duration_min = metrics['duration']
                current_route.total_wait_time_min = metrics['wait']
                current_route.total_load_kg = total_kg
                current_route.total_load_cbm = total_cbm
                return current_route # Done
                
    return current_route

def _quick_sim(data, vehicle, nodes):
    # Simulation đơn giản để check Time feasibility khi đổi xe
    curr_time = data.time_windows[0][0]
    total_dist = 0
    total_wait = 0
    prev = 0
    v_id = vehicle.type_id
    
    for node in nodes:
        dist = data.dist_matrix[prev, node]
        t_travel = data.get_travel_time(prev, node, v_id)
        curr_time += t_travel
        total_dist += dist
        
        start, end = data.time_windows[node]
        if curr_time > end: return False, None
        if curr_time < start:
            total_wait += (start - curr_time)
            curr_time = start
        curr_time += data.service_times[node]
        prev = node
        
    # Return depot
    curr_time += data.get_travel_time(prev, 0, v_id)
    total_dist += data.dist_matrix[prev, 0]
    
    if curr_time > data.time_windows[0][1]: return False, None
    dur = curr_time - data.time_windows[0][0]
    if dur > data.max_route_duration: return False, None
    
    return True, {'dist': total_dist, 'duration': dur, 'wait': total_wait}

def update_single_route_metrics(route: Route, data: ProblemData):
    """
    Update metrics VÀ thử Downgrade xe nếu tải trọng giảm.
    """
    if not route.node_sequence:
        route.total_dist_meters = 0
        return

    # 1. Update Metrics cơ bản với xe hiện tại (để có Load chính xác)
    # (Logic simulation cũ...)
    # ... Tôi gọi lại hàm _quick_sim cho xe hiện tại để update
    _, metrics = _quick_sim(data, route.vehicle_type, route.node_sequence)
    if metrics:
        route.total_dist_meters = metrics['dist']
        route.total_duration_min = metrics['duration']
        route.total_wait_time_min = metrics['wait']
        route.total_load_kg = sum(data.demands_kg[n] for n in route.node_sequence)
        route.total_load_cbm = sum(data.demands_cbm[n] for n in route.node_sequence)
    
    # 2. [NEW] Try Downgrade Vehicle
    _find_cheapest_feasible_vehicle(data, route)

def update_destroyed_state(state: RvrpState, data: ProblemData) -> RvrpState:
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

def create_random_customer_removal_operator(data: ProblemData, ratio: float = None):
    defaults = get_destroy_params(data.num_nodes)["random_removal"]

    def random_removal(state: RvrpState, rng, data: ProblemData):
        """
        Removes a number of randomly selected customers from the passed-in solution.
        """
        destroyed = state.copy()
        n_cust = data.num_nodes - 1
        
        if ratio is not None:
            num_remove = int(n_cust * ratio)
        else:
            num_remove = rng.integers(defaults["min"], defaults["max"] + 1)
        
        num_remove = max(1, min(num_remove, n_cust))
        
        # Flatten all assigned nodes
        assigned_nodes = [n for r in destroyed.routes for n in r.node_sequence]
        if not assigned_nodes: return destroyed

        actual_remove = min(num_remove, len(assigned_nodes))
        targets = rng.choice(assigned_nodes, actual_remove, replace=False)

        for node in targets:
            route = _find_route_containing_node(destroyed, node)
            if route:
                route.node_sequence.remove(node)
                destroyed.unassigned.append(node)
        
        return update_destroyed_state(destroyed, data) # UPDATE HERE
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