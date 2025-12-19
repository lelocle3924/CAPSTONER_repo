# file: alns/vrp4ppo/destroyopt.py

import numpy as np
from core.data_structures import RvrpState, ProblemData, Route
from .initial import neighbors

#ver 2: có thêm ratio vào từng operator, giúp xác định số lượng xóa
# có đổi lại cho đúng cấu trúc dataclass Route

def get_destroy_params(num_nodes):
    """
    Giữ lại để tương thích ngược (Backward Compatibility)
    nếu không truyền ratio cụ thể.
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
        # Các tham số khác...
    }

# --- HELPER FUNCTIONS (Giữ nguyên logic xử lý Object Route) ---

def _find_route_containing_node(state: RvrpState, node: int) -> Route:
    for route in state.routes:
        if node in route.node_sequence:
            return route
    return None

def _cleanup_empty_routes(state: RvrpState) -> RvrpState:
    state.routes = [r for r in state.routes if len(r.node_sequence) > 0]
    return state

def _calc_removal_gain(route: Route, node: int, data: ProblemData) -> float:
    seq = route.node_sequence
    idx = seq.index(node)
    prev_node = 0 if idx == 0 else seq[idx - 1]
    next_node = 0 if idx == len(seq) - 1 else seq[idx + 1]
    dist = data.dist_matrix
    return dist[prev_node, node] + dist[node, next_node] - dist[prev_node, next_node]

# --- MODIFIED OPERATORS (With Ratio) ---

def create_random_customer_removal_operator(data: ProblemData, ratio: float = None):
    # Load default params 1 lần để dùng nếu ratio=None
    defaults = get_destroy_params(data.num_nodes)["random_removal"]

    def random_removal(state: RvrpState, rng):
        destroyed = state.copy()
        n_cust = data.num_nodes - 1
        
        # LOGIC MỚI: Xác định số lượng dựa trên ratio
        if ratio is not None:
            num_remove = int(n_cust * ratio)
        else:
            num_remove = rng.integers(defaults["min"], defaults["max"] + 1)
            
        num_remove = max(1, min(num_remove, n_cust)) # Safety check

        # Lấy danh sách các node đang được gán
        assigned_nodes = [n for r in destroyed.routes for n in r.node_sequence]
        if not assigned_nodes: return destroyed

        # Nếu số lượng cần xóa > số node đang có, thì xóa hết số đang có
        actual_remove = min(num_remove, len(assigned_nodes))
        
        targets = rng.choice(assigned_nodes, actual_remove, replace=False)

        for node in targets:
            route = _find_route_containing_node(destroyed, node)
            if route:
                route.node_sequence.remove(node)
                destroyed.unassigned.append(node)
        
        return _cleanup_empty_routes(destroyed)
    return random_removal

def create_string_removal_operator(data: ProblemData, ratio: float = None):
    """
    String Removal (Path Removal):
    Chọn một tâm điểm ngẫu nhiên, sau đó xóa các đoạn tuyến đường (strings) 
    đi qua các điểm lân cận của tâm điểm đó.
    Mục tiêu: Xóa sạch một khu vực địa lý (Regional Removal).
    """
    defaults = get_destroy_params(data.num_nodes)["string_removal"]
    
    # Giới hạn độ dài chuỗi tối đa mặc định nếu không tính toán
    MAX_STRING_SIZE = defaults.get("max_string_size", 5)

    def string_removal(state: RvrpState, rng):
        destroyed = state.copy()
        n_cust = data.num_nodes - 1
        
        # 1. Xác định số lượng cần xóa
        if ratio is not None:
            target_remove = int(n_cust * ratio)
        else:
            # Logic cũ: dựa trên max_string_removals
            target_remove = rng.integers(2, max(5, int(n_cust * 0.15)))
            
        target_remove = max(1, min(target_remove, n_cust))
        
        # 2. Chọn tâm bão (Center Node)
        center_node = rng.integers(1, data.num_nodes)
        
        # Lấy danh sách hàng xóm gần tâm nhất
        # data.dist_matrix đã có sẵn, dùng hàm neighbors trong initial.py
        neighbor_nodes = neighbors(data.dist_matrix, center_node)
        
        removed_count = 0
        visited_routes = set()

        for customer in neighbor_nodes:
            if removed_count >= target_remove:
                break
                
            # Bỏ qua nếu khách này đã nằm trong unassigned (do lần lặp trước đã xóa)
            if customer in destroyed.unassigned:
                continue
                
            route = _find_route_containing_node(destroyed, customer)
            if not route or id(route) in visited_routes:
                continue
            
            # 3. Xóa một chuỗi xung quanh customer này trong route
            seq = route.node_sequence
            if not seq: continue
            
            cust_idx = seq.index(customer)
            
            # Random độ dài chuỗi cần xóa (từ 1 đến MAX_STRING_SIZE)
            # Hoặc tính dựa trên target còn lại
            remaining_quota = target_remove - removed_count
            string_len = rng.integers(1, min(remaining_quota, MAX_STRING_SIZE) + 1)
            
            # Xác định phạm vi xóa [start, end)
            # Cố gắng để customer nằm giữa chuỗi
            start = max(0, cust_idx - string_len // 2)
            end = min(len(seq), start + string_len)
            
            # Thực hiện xóa
            nodes_to_remove = seq[start:end]
            
            # Remove from list (phải cẩn thận xóa từ list gốc)
            # Cách an toàn: Tạo list mới không chứa các phần tử này
            route.node_sequence = [x for x in seq if x not in nodes_to_remove]
            
            destroyed.unassigned.extend(nodes_to_remove)
            removed_count += len(nodes_to_remove)
            
        return _cleanup_empty_routes(destroyed)

    return string_removal

def create_worst_removal_operator(data: ProblemData, ratio: float = None):
    defaults = get_destroy_params(data.num_nodes)["worst_removal"]

    def worst_removal(state: RvrpState, rng):
        destroyed = state.copy()
        n_cust = data.num_nodes - 1
        
        # LOGIC MỚI
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
            
            # Sort giảm dần (Gain cao nhất -> Xóa tiết kiệm nhất/Tệ nhất)
            costs.sort(key=lambda x: x[0], reverse=True)
            
            # Randomized Greedy selection (Ropke's formula)
            # p càng lớn càng deterministic (chọn đúng thằng tệ nhất)
            # p nhỏ thì random nhiều hơn
            p = 6 
            idx = int(len(costs) * (rng.random()**p))
            idx = min(idx, len(costs)-1)
            
            _, target_node, target_route = costs[idx]
            
            target_route.node_sequence.remove(target_node)
            destroyed.unassigned.append(target_node)
            
        return _cleanup_empty_routes(destroyed)
    return worst_removal

def create_random_route_removal_operator(ratio: float = None):
    """
    Với Route Removal: ratio sẽ được hiểu là % số lượng Route cần xóa
    """
    def route_removal(state: RvrpState, rng):
        destroyed = state.copy()
        if not destroyed.routes: return destroyed
        
        num_routes = len(destroyed.routes)
        
        # LOGIC MỚI
        if ratio is not None:
            count = int(num_routes * ratio)
        else:
            # Default cũ: Xóa 1 route ngẫu nhiên (hoặc logic cũ của thầy)
            # Ở đây em để mặc định là xóa 1 route cho an toàn
            count = 1
            
        count = max(1, min(count, num_routes))
        
        # Chọn ngẫu nhiên các route indices để xóa
        # Lưu ý: Khi pop index, thứ tự sẽ thay đổi, nên ta sort index giảm dần để pop từ dưới lên
        indices_to_remove = rng.choice(num_routes, count, replace=False)
        indices_to_remove.sort()
        
        # Pop từ dưới lên để không ảnh hưởng index của các phần tử phía trước
        for idx in reversed(indices_to_remove):
            removed_route = destroyed.routes.pop(idx)
            destroyed.unassigned.extend(removed_route.node_sequence)
        
        return destroyed
    return route_removal

def create_related_removal_operator(data: ProblemData, ratio: float = None):
    # (Shaw Removal)
    def calculate_relatedness(i, j, dist_matrix):
        dist = dist_matrix[i, j]
        # Có thể thêm logic same_route nếu cần, nhưng đơn giản hóa dùng khoảng cách
        return 1.0 / (dist + 0.01)

    def related_removal(state: RvrpState, rng):
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
                rel = calculate_relatedness(ref_node, cand, data.dist_matrix)
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

        return _cleanup_empty_routes(destroyed)
    return related_removal

def create_sequence_removal_operator(data: ProblemData, ratio: float = None):
    """
    Sequence Removal:
    Remove contiguous segments in a random route.
    """
    def sequence_removal(state: RvrpState, rng):
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

        return _cleanup_empty_routes(destroyed)

    return sequence_removal