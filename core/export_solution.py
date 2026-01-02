import pandas as pd
import folium
import os
import random
import requests
import time
import polyline

def get_distinct_colors(n):
    return [f'#{random.randint(50,200):02x}{random.randint(50,200):02x}{random.randint(50,200):02x}' for _ in range(n)]

def format_minutes_to_time(minutes):
    if minutes is None: return ""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    days = hours // 24
    display_hours = hours % 24
    if days > 0:
        return f"{display_hours:02d}:{mins:02d} (+{days}d)"
    return f"{display_hours:02d}:{mins:02d}"

def get_route_geometry_from_osrm(points, max_retries=3):
    """Lấy geometry tuyến đường từ OSRM API."""
    if len(points) < 2: return None
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in points])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=polyline"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == 'Ok' and data['routes']:
                    return data['routes'][0]['geometry']
            elif response.status_code == 429:
                time.sleep(2 * (attempt + 1))
        except requests.exceptions.RequestException:
            time.sleep(2 * (attempt + 1))
    return None

def calculate_node_metrics(data, from_node, to_node, arrival_time, time_window, vehicle_id):
    """
    Tính toán chi phí và khoảng cách cho chặng.
    Trả về: (Operating Cost, Penalty Cost, Distance KM)
    """
    dist_km = data['distance_matrix'][from_node][to_node] / 1000.0
    time_h = data['time_matrix'][from_node][to_node] / 60.0
    dwell_h = data['dwell_times'][from_node] / 60.0
    
    c_km = data['vehicle_costs_per_km'][vehicle_id]
    c_h = data['vehicle_costs_per_h'][vehicle_id]
    
    # 1. Chi phí vận hành
    op_cost = (dist_km * c_km) + ((time_h + dwell_h) * c_h)
    
    # 2. Chi phí phạt
    desired_end = max(w[1] for w in time_window) if time_window else 1440*3
    late_minutes = max(0, arrival_time - desired_end)
    LATE_PENALTY_COST = 5000 
    penalty_cost = late_minutes * LATE_PENALTY_COST
    
    return int(op_cost), int(penalty_cost), dist_km

def get_allowed_truck_names(data, node_index):
    """
    Lấy danh sách tên các loại xe được phép, trả về định dạng {Type1, Type2}.
    """
    if 'allowed_vehicles' not in data:
        return "{All}"
    
    allowed_ids = data['allowed_vehicles'][node_index]
    # Nếu danh sách ID bằng tổng số xe -> Cho phép tất cả
    if len(allowed_ids) == data['num_vehicles']:
        return "{All}"
    
    # Chuyển ID sang Tên (Lấy unique vì nhiều xe cùng loại)
    # Ví dụ: ID 0,1,2 đều là '4w' -> chỉ lấy '4w' một lần
    allowed_names = sorted(list(set([data['vehicle_names'][i] for i in allowed_ids])))
    
    # Format lại thành chuỗi {Name1, Name2}
    return "{" + ", ".join(allowed_names) + "}"


def export_solution_to_csv_and_map(data, manager, routing, solution, depot_id, time_dimension, kgm_dimension, cbm_dimension, delivery_date, output_dir="results"):
    print(">>> Đang trích xuất và tính toán KPI...")
    os.makedirs(output_dir, exist_ok=True)
    
    detailed_results = []
    summary_results = [] # Danh sách chứa dữ liệu tổng hợp
    vehicle_routes = {}
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue
            
        # Biến tạm cho từng Route
        route_nodes = []
        route_poly = []
        route_sequence_ids = [] # Dùng để tạo chuỗi Depot -> A -> B
        
        total_op_cost = 0
        total_penalty_cost = 0
        total_distance_km = 0
        
        # --- LOOP QUA CÁC ĐIỂM DỪNG ---
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            next_node_index = manager.IndexToNode(next_index)

            time_var = time_dimension.CumulVar(index)
            eta_min = solution.Min(time_var)
            
            # Tính toán Cost & Distance
            windows_list = data['time_windows'][node_index]
            op_cost, pen_cost, dist_km = calculate_node_metrics(
                data, node_index, next_node_index, eta_min, windows_list, vehicle_id
            )
            
            total_op_cost += op_cost
            total_penalty_cost += pen_cost
            total_distance_km += dist_km
            
            # Lưu ID vào chuỗi hành trình
            loc_id = str(data['locations'][node_index])
            route_sequence_ids.append(loc_id)

            # Format Time Window
            window_strs = [f"{format_minutes_to_time(s)}-{format_minutes_to_time(e)}" for s, e in windows_list]
            allowed_trucks_str = get_allowed_truck_names(data, node_index)
            route_nodes.append({
                'StopOrder': len(route_nodes) + 1,
                'LocationID': loc_id,
                'Latitude': data['node_coordinates'][node_index][0],
                'Longitude': data['node_coordinates'][node_index][1],
                'Allowed_Trucks': allowed_trucks_str,
                'ETA': format_minutes_to_time(eta_min),
                'TimeWindow': " / ".join(window_strs),
                'Demand_KGM': data['demands_kgm'][node_index],
                'Demand_CBM': data['demands_cbm'][node_index] / 1000.0,
                'Segment_Dist_KM': round(dist_km, 2),
                'Segment_Cost': op_cost + pen_cost
            })
            
            route_poly.append((data['node_coordinates'][node_index][0], data['node_coordinates'][node_index][1]))
            index = next_index
            
        # --- XỬ LÝ ĐIỂM CUỐI (DEPOT) ---
        node_index = manager.IndexToNode(index)
        time_var = time_dimension.CumulVar(index)
        eta_min = solution.Min(time_var)
        route_sequence_ids.append(str(data['locations'][node_index])) # Thêm Depot vào cuối chuỗi
        
        route_nodes.append({
            'StopOrder': len(route_nodes) + 1,
            'LocationID': data['locations'][node_index],
            'Latitude': data['node_coordinates'][node_index][0],
            'Longitude': data['node_coordinates'][node_index][1],
            
            'Allowed_Trucks': "{All}",
            'ETA': format_minutes_to_time(eta_min),
            'TimeWindow': "Full Day",
            'Demand_KGM': 0, 'Demand_CBM': 0.0,
            'Segment_Dist_KM': 0, 'Segment_Cost': 0
        })
        route_poly.append((data['node_coordinates'][node_index][0], data['node_coordinates'][node_index][1]))

        # --- TỔNG HỢP DỮ LIỆU ---
        # 1. Lấy Geometry
        encoded_geometry = get_route_geometry_from_osrm(route_poly)
        
        # 2. Tính tổng tải trọng
        sum_kgm = sum(n['Demand_KGM'] for n in route_nodes)
        sum_cbm = sum(n['Demand_CBM'] for n in route_nodes)
        
        # 3. Tính chi phí tổng
        fixed_cost = data['vehicle_fixed_costs'][vehicle_id]
        total_op_cost += fixed_cost # Cộng phí mở cửa xe
        final_total_cost = total_op_cost + total_penalty_cost
        
        # 4. Tính độ hiệu quả (Utilization)
        cap_kgm = data['vehicle_capacities_kgm'][vehicle_id]
        cap_cbm = data['vehicle_capacities_cbm'][vehicle_id] / 1000.0
        util_kgm = round((sum_kgm / cap_kgm) * 100, 1) if cap_kgm > 0 else 0
        util_cbm = round((sum_cbm / cap_cbm) * 100, 1) if cap_cbm > 0 else 0
        
        # 5. Tạo chuỗi tuyến đường đơn giản
        simple_route_str = " -> ".join(route_sequence_ids)
        
        vehicle_name = data['vehicle_names'][vehicle_id]

        # --- LƯU VÀO SUMMARY SHEET ---
        summary_results.append({
            'RouteID': f'R_{delivery_date}_{vehicle_id}',
            'VehicleID': vehicle_id,
            'VehicleType': vehicle_name,
            'Num_Stops': len(route_nodes) - 2, # Trừ 2 lần Depot
            'Total_Distance_KM': round(total_distance_km, 2),
            'Total_Duration_Min': eta_min, # Thời gian về đến kho
            'Total_Load_KGM': sum_kgm,
            'Total_Load_CBM': sum_cbm,
            'Capacity_KGM': cap_kgm,
            'Capacity_CBM': cap_cbm,
            'Util_KGM_%': util_kgm,
            'Util_CBM_%': util_cbm,
            'Fixed_Cost': fixed_cost,
            'Variable_Cost': total_op_cost - fixed_cost,
            'Penalty_Cost': total_penalty_cost,
            'Total_Trip_Cost': final_total_cost,
            'Cost_Per_KG': round(final_total_cost / sum_kgm, 2) if sum_kgm > 0 else 0,
            'Simple_Route_Sequence': simple_route_str
        })

        # --- LƯU VÀO DETAILED SHEET ---
        curr_kgm = sum_kgm
        for node in route_nodes:
            node['Load_On_Board'] = curr_kgm
            curr_kgm -= node['Demand_KGM']
            
            detailed_results.append({
                'RouteID': f'R_{delivery_date}_{vehicle_id}',
                'VehicleType': vehicle_name,
                'Route_Total_Cost': final_total_cost, 
                **node,
            })
            
        vehicle_routes[vehicle_id] = {'nodes': route_nodes, 'geometry': encoded_geometry}

    # --- XUẤT RA FILE EXCEL (2 SHEETS) ---
    if not summary_results:
        print("Không có lời giải.")
        return

    excel_filename = f"solution_{depot_id}_{delivery_date}.xlsx"
    excel_filepath = os.path.join(output_dir, excel_filename)
    
    df_summary = pd.DataFrame(summary_results)
    df_details = pd.DataFrame(detailed_results)
    
    # Sắp xếp cột cho sheet Summary đẹp hơn
    sum_cols = ['RouteID', 'VehicleType','Allowed_Trucks' ,'Num_Stops' , 'Total_Distance_KM', 
                'Total_Load_KGM','Total_Load_CBM','Capacity_KGM' ,'Capacity_CBM', 'Util_KGM_%', 'Util_CBM_%', 'Total_Trip_Cost', 
                'Fixed_Cost', 'Variable_Cost', 'Penalty_Cost', 
                'Cost_Per_KG', 'Simple_Route_Sequence']
    final_sum_cols = [c for c in sum_cols if c in df_summary.columns] + [c for c in df_summary.columns if c not in sum_cols]
    detail_cols = [
        'RouteID', 'VehicleType', 'StopOrder', 'LocationID', 
        'Allowed_Trucks', 'ETA', 'TimeWindow', 
        'Demand_KGM', 'Load_On_Board_KGM',
        'Segment_Dist_KM', 'Segment_Cost'
    ]
    final_detail_cols = [c for c in detail_cols if c in df_details.columns] + [c for c in df_details.columns if c not in detail_cols]
    with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
        df_summary[final_sum_cols].to_excel(writer, sheet_name='Route_Summary', index=False)
        df_details[final_detail_cols].to_excel(writer, sheet_name='Stop_Details', index=False)
        
    print(f">>> Đã xuất file Excel tại: {excel_filepath}")
    
    # --- VẼ MAP (Giữ nguyên logic cũ) ---
    map_filename = f"map_{depot_id}_{delivery_date}.html"
    map_filepath = os.path.join(output_dir, map_filename)
    m = folium.Map(location=[df_details['Latitude'].mean(), df_details['Longitude'].mean()], zoom_start=11)
    colors = get_distinct_colors(len(vehicle_routes))
    
    for i, (v_id, route_data) in enumerate(vehicle_routes.items()):
        color = colors[i]
        if route_data['geometry']:
            folium.PolyLine(polyline.decode(route_data['geometry']), color=color, weight=3, opacity=0.8).add_to(m)
        else:
            points = [(n['Latitude'], n['Longitude']) for n in route_data['nodes']]
            folium.PolyLine(points, color=color, weight=2.5, opacity=0.5, dash_array='5, 5').add_to(m)
            
        for node in route_data['nodes']:
            if node['LocationID'] != 'Depot':
                folium.CircleMarker(
                    location=(node['Latitude'], node['Longitude']),
                    radius=4, color=color, fill=True, fill_color='white',
                    popup=f"Stop {node['StopOrder']}: {node['LocationID']}"
                ).add_to(m)
    
    m.save(map_filepath)
    print(f">>> Đã lưu bản đồ: {map_filepath}")