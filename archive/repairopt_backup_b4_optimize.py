# def create_greedy_repair_operator(data: ProblemData):
#     def greedy_repair(state: RvrpState, rng, data: ProblemData):
#         rng.shuffle(state.unassigned)
#         while state.unassigned:
#             customer = state.unassigned.pop()
#             best_diff = float('inf')
#             best_r_idx = -1
#             best_pos = -1
#             best_metrics = None
#             best_vehicle = None 
            
#             # 1. Try insert into existing routes
#             for r_idx, route in enumerate(state.routes):
#                 prev_cost = route.cost 
                
#                 for i in range(len(route.node_sequence) + 1):
#                     new_abs_cost, metrics, new_veh = calculate_insertion_cost(data, route, customer, i)
                    
#                     if metrics:
#                         diff = new_abs_cost - prev_cost
#                         if diff < best_diff:
#                             best_diff = diff
#                             best_r_idx = r_idx
#                             best_pos = i
#                             best_metrics = metrics
#                             best_vehicle = new_veh
                            
#             # 2. Try create new route
#             new_route = _try_create_new_route(data, customer)
#             if new_route:
#                 new_cost = new_route.cost
#                 if new_cost < best_diff:
#                     best_diff = new_cost
#                     best_r_idx = -1 
            
#             # 3. Apply Best Move
#             if best_r_idx != -1:
#                 target_route = state.routes[best_r_idx]
#                 target_route.node_sequence.insert(best_pos, customer)
#                 # Update attributes
#                 target_route.total_dist_meters = best_metrics["total_dist_meters"]
#                 target_route.total_duration_min = best_metrics["total_duration_min"]
#                 target_route.total_wait_time_min = best_metrics["total_wait_time_min"]
#                 target_route.total_load_kg = best_metrics["total_load_kg"]
#                 target_route.total_load_cbm = best_metrics["total_load_cbm"]
                
#                 # [CRITICAL] Update Vehicle Type if Upgraded
#                 if best_vehicle.type_id != target_route.vehicle_type.type_id:
#                     target_route.vehicle_type = best_vehicle
                    
#             elif new_route:
#                 state.routes.append(new_route)
#             else:
#                 pass # Infeasible to serve this customer -> Leave unassigned
                
#         return state
#     return greedy_repair

# def create_regret_repair_operator(data: ProblemData, k: int = 2):
#     """
#     Regret Repair: Tính nuối tiếc nếu không chọn lựa chọn tốt nhất.
#     Logic UPGRADE VEHICLE đã được tích hợp.
#     """
#     def regret_repair(state: RvrpState, rng, data: ProblemData):
#         while state.unassigned:
#             # List chứa thông tin regret của từng customer
#             # (regret_val, customer, r_idx, pos, metrics, vehicle)
#             candidates_regret = []
            
#             for customer in state.unassigned:
#                 # Tìm tất cả feasible insertions cho customer này
#                 valid_insertions = [] # (cost_increase, r_idx, pos, metrics, vehicle)
                
#                 # 1. Quét Existing Routes
#                 for r_idx, route in enumerate(state.routes):
#                     prev_cost = route.cost
                    
#                     local_best_diff = float('inf')
#                     local_best_entry = None # (diff, pos, metrics, veh)
                    
#                     for i in range(len(route.node_sequence) + 1):
#                         new_cost, metrics, new_veh = calculate_insertion_cost(data, route, customer, i)
#                         if metrics:
#                             diff = new_cost - prev_cost
#                             if diff < local_best_diff:
#                                 local_best_diff = diff
#                                 local_best_entry = (diff, i, metrics, new_veh)
                    
#                     if local_best_entry:
#                         valid_insertions.append((local_best_entry[0], r_idx, local_best_entry[1], local_best_entry[2], local_best_entry[3]))
                
#                 # 2. Quét New Route Option
#                 new_route = _try_create_new_route(data, customer)
#                 if new_route:
#                     # New route metrics
#                     m = {
#                         "total_dist_meters": new_route.total_dist_meters,
#                         "total_duration_min": new_route.total_duration_min,
#                         "total_wait_time_min": new_route.total_wait_time_min,
#                         "total_load_kg": new_route.total_load_kg,
#                         "total_load_cbm": new_route.total_load_cbm
#                     }
#                     valid_insertions.append((new_route.cost, -1, 0, m, new_route.vehicle_type))
                
#                 # 3. Tính Regret
#                 if not valid_insertions: continue # Khách này ko chèn được đâu cả
                
#                 valid_insertions.sort(key=lambda x: x[0]) # Sort theo cost tăng dần
                
#                 best = valid_insertions[0]
#                 if len(valid_insertions) >= k:
#                     second = valid_insertions[k-1]
#                     regret_val = second[0] - best[0]
#                 else:
#                     regret_val = float('inf') # Must insert now
                    
#                 candidates_regret.append({
#                     "regret": regret_val,
#                     "customer": customer,
#                     "r_idx": best[1],
#                     "pos": best[2],
#                     "metrics": best[3],
#                     "vehicle": best[4]
#                 })
            
#             if not candidates_regret: break
            
#             # Chọn Max Regret
#             candidates_regret.sort(key=lambda x: x["regret"], reverse=True)
#             winner = candidates_regret[0]
            
#             # Apply
#             cust = winner["customer"]
#             state.unassigned.remove(cust)
            
#             if winner["r_idx"] != -1:
#                 target_route = state.routes[winner["r_idx"]]
#                 target_route.node_sequence.insert(winner["pos"], cust)
#                 m = winner["metrics"]
#                 target_route.total_dist_meters = m["total_dist_meters"]
#                 target_route.total_duration_min = m["total_duration_min"]
#                 target_route.total_wait_time_min = m["total_wait_time_min"]
#                 target_route.total_load_kg = m["total_load_kg"]
#                 target_route.total_load_cbm = m["total_load_cbm"]
                
#                 # [CRITICAL] Upgrade Vehicle
#                 best_veh = winner["vehicle"]
#                 if best_veh.type_id != target_route.vehicle_type.type_id:
#                     target_route.vehicle_type = best_veh
#             else:
#                 new_route = _try_create_new_route(data, cust) # Re-create object
#                 if new_route: state.routes.append(new_route)

#         return state
#     return regret_repair

#def create_criticality_repair_operator(data: ProblemData):
#     """
#     Repair dựa trên độ ưu tiên (Criticality).
#     Logic UPGRADE VEHICLE đã được tích hợp.
#     """
#     depot_dists = data.dist_matrix[0, :]
#     min_tw = 30.0 
    
#     def get_importance(node_idx):
#         norm_dem = data.demands_kg[node_idx] / 1000.0
#         norm_dist = depot_dists[node_idx] / 10000.0
#         tw_len = data.time_windows[node_idx][1] - data.time_windows[node_idx][0]
#         norm_tw = 1.0 / (max(tw_len, min_tw) / 60.0)
#         return norm_dem + norm_dist + norm_tw

#     def criticality_repair(state: RvrpState, rng, data: ProblemData):
#         # Sort unassigned: Khó nhất xếp cuối để pop ra đầu
#         state.unassigned.sort(key=get_importance) 

#         while state.unassigned:
#             customer = state.unassigned.pop() 
            
#             best_diff = float('inf')
#             best_r_idx = -1
#             best_pos = -1
#             best_metrics = None
#             best_vehicle = None
            
#             # 1. Existing Routes
#             for r_idx, route in enumerate(state.routes):
#                 prev_cost = route.cost 
                
#                 for i in range(len(route.node_sequence) + 1):
#                     new_abs_cost, metrics, new_veh = calculate_insertion_cost(data, route, customer, i)
#                     if metrics:
#                         diff = new_abs_cost - prev_cost
#                         if diff < best_diff:
#                             best_diff = diff
#                             best_r_idx = r_idx
#                             best_pos = i
#                             best_metrics = metrics
#                             best_vehicle = new_veh
                            
#             # 2. New Route
#             new_route = _try_create_new_route(data, customer)
#             if new_route:
#                 if new_route.cost < best_diff:
#                     best_diff = new_route.cost
#                     best_r_idx = -1
                    
#             # 3. Apply
#             if best_r_idx != -1:
#                 target_route = state.routes[best_r_idx]
#                 target_route.node_sequence.insert(best_pos, customer)
#                 target_route.total_dist_meters = best_metrics["total_dist_meters"]
#                 target_route.total_duration_min = best_metrics["total_duration_min"]
#                 target_route.total_wait_time_min = best_metrics["total_wait_time_min"]
#                 target_route.total_load_kg = best_metrics["total_load_kg"]
#                 target_route.total_load_cbm = best_metrics["total_load_cbm"]
                
#                 # [CRITICAL] Update Vehicle
#                 if best_vehicle.type_id != target_route.vehicle_type.type_id:
#                     target_route.vehicle_type = best_vehicle
#             elif new_route:
#                 state.routes.append(new_route)

#         return state
#     return criticality_repair