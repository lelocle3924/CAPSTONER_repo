import numpy as np
import math
from core.data_structures import RvrpState, Route, ProblemData, VehicleType

def calculate_route_metrics(route_nodes: list[int], vehicle: VehicleType, data: ProblemData):
    """
    Hàm helper tính toán metrics cho 1 chuỗi node cụ thể với loại xe cụ thể.
    Trả về Dict metrics hoặc None nếu Infeasible.
    """
    total_dist = 0.0
    total_load_kg = 0.0
    total_load_cbm = 0.0
    
    current_time = 0.0 # Bắt đầu từ 0 (Depot Open)
    total_wait = 0.0
    
    # 1. Tại Depot (Start)
    prev_node = 0 
    
    # Depot Time Window: [Open, Close]
    depot_open = data.time_windows[0][0]
    depot_close = data.time_windows[0][1]
    
    current_time = max(current_time, depot_open)
    #current_time = 0

    for node in route_nodes:
        # --- TRAVEL ---
        dist = data.dist_matrix[prev_node, node]
        
        # [UPDATED] Lấy thời gian từ Super Matrix dựa trên Vehicle Type ID
        travel_time = data.get_travel_time(prev_node, node, vehicle.type_id)
        
        total_dist += dist
        current_time += travel_time
        
        # --- TIME WINDOW & WAIT ---
        start_window = data.time_windows[node][0]
        end_window = data.time_windows[node][1]
        
        if current_time > end_window:
            return None, "Time Window Violation" # Đến muộn
        
        if current_time < start_window:
            wait = start_window - current_time
            total_wait += wait
            current_time = start_window # Chờ đến giờ mở cửa
            
        # --- SERVICE ---
        service = data.service_times[node]
        current_time += service
        
        # --- CAPACITY ---
        total_load_kg += data.demands_kg[node]
        total_load_cbm += data.demands_cbm[node]
        
        prev_node = node

    # --- RETURN TO DEPOT ---
    dist_back = data.dist_matrix[prev_node, 0]
    time_back = data.get_travel_time(prev_node, 0, vehicle.type_id)
    
    total_dist += dist_back
    current_time += time_back
    
    # Check Depot Closing
    if current_time > depot_close:
        return None, "Depot Closing Violation"
        
    # Check Max Duration
    total_duration = current_time - depot_open
    if total_duration > data.max_route_duration:
        return None, "Max Duration Violation"
        
    # Check Capacity
    if total_load_kg > vehicle.capacity_kg or total_load_cbm > vehicle.capacity_cbm:
        return None, "Capacity Violation"
        
    return {
        "dist": total_dist,
        "duration": total_duration,
        "wait": total_wait,
        "load_kg": total_load_kg,
        "load_cbm": total_load_cbm
    }, "OK"


def one_for_one(data: ProblemData) -> RvrpState:
    """
    Chiến thuật an toàn: Mỗi khách 1 xe.
    Cải tiến: Chọn xe nhỏ nhất vừa đủ tải (Best Fit) để tối ưu chi phí ban đầu.
    """
    routes = []
    unassigned = []

    # Sắp xếp xe từ bé đến lớn để tìm Best Fit
    sorted_fleet = sorted(data.vehicle_types, key=lambda v: v.capacity_kg)

    for customer_idx in range(1,len(data.node_ids)):
        allowed_ids = data.allowed_vehicles[customer_idx]
        
        best_route = None
        best_cost = float('inf')
        
        # Thử gán cho từng loại xe được phép, từ bé đến lớn
        found_vehicle = False
        
        for vehicle in sorted_fleet:
            # 1. Check Allowed
            if vehicle.type_id not in allowed_ids:
                continue
                
            # 2. Check Capacity Sơ bộ (để đỡ tính toán nặng nếu hàng quá to)
            if (data.demands_kg[customer_idx] > vehicle.capacity_kg or 
                data.demands_cbm[customer_idx] > vehicle.capacity_cbm):
                print("over capacity")
                continue
                
            # 3. Tính toán chi tiết (Time, Duration)
            metrics, status = calculate_route_metrics([customer_idx], vehicle, data)
            if metrics is not None:
                # Tính cost để so sánh (ưu tiên xe rẻ tiền)
                # Cost = Fixed + Var
                # Đơn vị dist là mét -> đổi ra km
                cost = (vehicle.fixed_cost + 
                        (metrics['dist'] / 1000.0) * vehicle.cost_per_km + 
                        (metrics['duration'] / 60.0) * vehicle.cost_per_hour)
                
                if cost < best_cost:
                    best_cost = cost
                    found_vehicle = True
                    
                    best_route = Route(
                        vehicle_type=vehicle,
                        node_sequence=[customer_idx],
                        total_dist_meters=metrics['dist'],
                        total_duration_min=metrics['duration'],
                        total_wait_time_min=metrics['wait'],
                        total_load_kg=metrics['load_kg'],
                        total_load_cbm=metrics['load_cbm'],
                        is_time_feasible=True,
                        is_capacity_feasible=True,
                        is_duration_feasible=True
                    )
                    break # Vì đã sort từ bé đến lớn, tìm thấy cái đầu tiên là cái rẻ nhất (thường là vậy) -> Break luôn
        
        if found_vehicle or best_route:
            routes.append(best_route)
        else:
            # Không xe nào chở được (do Time Window, Duration hoặc tải trọng)
            # print(f"Customer {customer_idx} Unassigned. Reason: Constraints")
            unassigned.append(customer_idx)

    return RvrpState(routes=routes, unassigned=unassigned)


# --- CÁC HÀM CŨ (LEGACY) ---
# Tôi comment lại để tránh gọi nhầm. Khi nào sửa xong logic Heterogeneous cho bọn này thì mở ra.

def neighbors(dist_matrix, customer_idx):
    locations = np.argsort(dist_matrix[customer_idx])
    return locations[locations != 0]

# def nearest_neighbor(dimension, demand, capacity, edge_weight):
#     """
#     Build a solution by iteratively constructing routes, where the nearest
#     customer is added until the route has met the vehicle capacity limit.
#     """
#     routes = []
#     unvisited = set(range(1, dimension))

#     while unvisited:
#         route = [0]  # Start at the depot
#         route_demands = 0

#         while unvisited:
#             # Add the nearest unvisited customer to the route till max capacity
#             current = route[-1]
#             nearest = [nb for nb in neighbors(edge_weight, current) if nb in unvisited][
#                 0
#             ]

#             if route_demands + demand[nearest] > capacity:
#                 break

#             route.append(nearest)
#             unvisited.remove(nearest)
#             route_demands += demand[nearest]

#         customers = route[1:]  # Remove the depot
#         routes.append(customers)

#     return CvrpState(routes)


# def neighbors_tw(travel_times, time_windows, current_node, current_time):
#     """
#     Return feasible neighboring nodes of current node considering time window constraints

#     Args:
#         travel_times: Travel time matrix
#         time_windows: Time window constraints
#         current_node: Current node
#         current_time: Current time
#     """
#     # Get all locations sorted by distance
#     locations = np.argsort(travel_times[current_node])

#     # Filter out depot and nodes that don't satisfy time window
#     feasible = []
#     for next_node in locations[locations != 0]:
#         arrival_time = current_time + travel_times[current_node][next_node]
#         earliest, latest = time_windows[next_node]
#         # Wait if arrive early
#         if arrival_time < earliest:
#             arrival_time = earliest
#         # If can arrive before the latest arrival time, it's a feasible node
#         if arrival_time <= latest:
#             feasible.append(next_node)

#     return feasible


# def nearest_neighbor_tw(data):
#     """
#     Nearest neighbor construction algorithm considering time window constraints
#     """
#     dimension = data["dimension"]
#     demand = data["demand"]
#     capacity = data["capacity"]
#     travel_times = data["travel_times"]
#     time_windows = data["time_windows"]
#     service_times = data["service_times"]

#     routes = []
#     unvisited = set(range(1, dimension))

#     while unvisited:
#         route = [0]  # Start from depot
#         route_demands = 0
#         current_time = 0  # Current route time

#         while unvisited:
#             current = route[-1]

#             # Get all feasible neighboring nodes that satisfy time window constraints
#             feasible_neighbors = neighbors_tw(
#                 travel_times, time_windows, current, current_time
#             )

#             # Filter out visited nodes and nodes exceeding capacity
#             feasible_neighbors = [
#                 nb
#                 for nb in feasible_neighbors
#                 if nb in unvisited and route_demands + demand[nb] <= capacity
#             ]

#             if not feasible_neighbors:
#                 break

#             # Select the nearest feasible node
#             nearest = feasible_neighbors[0]

#             # Update arrival time
#             travel_time = travel_times[current][nearest]
#             arrival_time = current_time + travel_time

#             # Wait if arrive early
#             earliest = time_windows[nearest][0]
#             if arrival_time < earliest:
#                 arrival_time = earliest

#             # Update current time (add service time)
#             current_time = arrival_time + service_times[nearest]

#             # Check if can return to depot on time
#             if current_time + travel_times[nearest][0] > time_windows[0][1]:
#                 # Exceeds maximum travel time, abandon current customer
#                 break

#             # Update route
#             route.append(nearest)
#             unvisited.remove(nearest)
#             route_demands += demand[nearest]

#         # If route is not empty, add to routes
#         if len(route) > 1:
#             routes.append(route[1:])

#     return CvrptwState(routes, travel_times)


# def clarke_wright_tw(data):
#     """
#     VRPTW implementation of Clarke-Wright savings algorithm
#     """
#     dimension = data["dimension"]
#     demand = data["demand"]
#     capacity = data["capacity"]
#     travel_times = data["travel_times"]

#     # 1. Initialize: create independent route for each customer
#     routes = [[i] for i in range(1, dimension)]

#     # 2. Calculate savings values
#     savings = []
#     for i in range(1, dimension):
#         for j in range(i + 1, dimension):
#             # saving = cost(0,i) + cost(0,j) - cost(i,j)
#             saving = travel_times[0][i] + travel_times[0][j] - travel_times[i][j]
#             savings.append((saving, i, j))

#     # Sort by savings values in descending order
#     savings.sort(reverse=True)

#     # 3. Merge routes based on savings values
#     # Record route index for each customer
#     customer_route = {i: idx for idx, route in enumerate(routes) for i in route}

#     for saving, i, j in savings:
#         if saving <= 0:
#             break

#         # Find routes containing i and j
#         route_i = customer_route.get(i)
#         route_j = customer_route.get(j)

#         # If i and j are already in the same route or a point is not in any route
#         if route_i is None or route_j is None or route_i == route_j:
#             continue

#         # Get two routes
#         path_i = routes[route_i]
#         path_j = routes[route_j]

#         # Check if merged route satisfies capacity constraints
#         total_demand = sum(demand[k] for k in path_i + path_j)
#         if total_demand > capacity:
#             continue

#         # Try different merge methods and check time window constraints
#         merged = None
#         if (
#             path_i[-1] == i and path_j[0] == j
#         ):  # i at end of route i, j at start of route j
#             merged = path_i + path_j
#         elif (
#             path_i[0] == i and path_j[-1] == j
#         ):  # i at start of route i, j at end of route j
#             merged = path_j + path_i

#         if merged and is_time_feasible(merged, data):
#             # Update route
#             routes[route_i] = merged
#             routes[route_j] = []
#             # Update customer route indices
#             for customer in merged:
#                 customer_route[customer] = route_i
#             customer_route[j] = None

#     # Remove empty routes and return results
#     return CvrptwState([r for r in routes if r], travel_times)
