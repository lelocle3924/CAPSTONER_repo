import pandas as pd
import requests
import numpy as np
import os
import polyline
import folium
import time
from math import radians, cos, sin, asin, sqrt

#ver 4

class DistanceMatrixCalculator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.osrm_table_url = "http://router.project-osrm.org/table/v1/driving/"
        self.osrm_route_url = "http://router.project-osrm.org/route/v1/driving/"
        
        self.nodes = self._load_and_parse_nodes()
        self._cached_distance_array_meters = None 
        
        self.BATCH_SIZE = 60
        self.REQUEST_DELAY = 1.0 
        self.TIMEOUT = 30

    def _load_and_parse_nodes(self):
        try:
            df = pd.read_csv(self.file_path)
        except Exception as e:
            raise Exception(f"Input error: {e}")

        depot_info = df.iloc[0]
        depot_node = {
            'ID': 'Depot',
            'Lat': float(depot_info['DepotLat']),
            'Long': float(depot_info['DepotLong']),
            'Type': 'Depot'
        }

        unique_customers = df.groupby('Customer').agg({
            'CusLat': 'first',
            'CusLong': 'first'
        }).reset_index()

        customer_nodes = []
        for _, row in unique_customers.iterrows():
            customer_nodes.append({
                'ID': str(int(row['Customer'])),
                'Lat': float(row['CusLat']),
                'Long': float(row['CusLong']),
                'Type': 'Customer'
            })

        return [depot_node] + customer_nodes

    def _calculate_haversine_subset(self, src_nodes, dst_nodes):
        # FALLBACK
        matrix = np.zeros((len(src_nodes), len(dst_nodes)))
        for i, src in enumerate(src_nodes):
            for j, dst in enumerate(dst_nodes):
                lat1, lon1 = radians(src['Lat']), radians(src['Long'])
                lat2, lon2 = radians(dst['Lat']), radians(dst['Long'])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                # ROUTE FACTOR to be equivalent to real distance
                matrix[i][j] = c * 6371 * 1000 * 1.3 
        return matrix

    def _fetch_chunk_with_retry(self, src_nodes, dst_nodes, max_retries=3):
        all_chunk_nodes = src_nodes + dst_nodes
        coords_str = ";".join([f"{n['Long']},{n['Lat']}" for n in all_chunk_nodes])
        
        src_indices = ";".join([str(i) for i in range(len(src_nodes))])
        dst_indices = ";".join([str(i + len(src_nodes)) for i in range(len(dst_nodes))])
        
        request_url = f"{self.osrm_table_url}{coords_str}?sources={src_indices}&destinations={dst_indices}&annotations=distance"
        
        # RETRY
        for attempt in range(max_retries):
            try:
                time.sleep(self.REQUEST_DELAY)
                
                response = requests.get(request_url, timeout=self.TIMEOUT)
                
                if response.status_code == 200:
                    data = response.json()
                    return np.array(data['distances'])
                elif response.status_code == 429: # Too Many Requests
                    print(f"    [429] Server busy. Retrying in {2*(attempt+1)}s...")
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"    [API Error] Status {response.status_code}. Retrying...")
            
            except requests.exceptions.RequestException as e:
                print(f"    [Net Error] {e}. Attempt {attempt+1}/{max_retries}")
                time.sleep(2 * (attempt + 1))
        
        return None

    def _fetch_full_matrix_batched(self):
        n = len(self.nodes)
        full_matrix = np.zeros((n, n))
        
        print(f"  > Starting batch processing for {n} nodes...")
        print(f"  > Config: Batch Size={self.BATCH_SIZE}, Delay={self.REQUEST_DELAY}s")
        
        for i in range(0, n, self.BATCH_SIZE):
            src_chunk = self.nodes[i : i + self.BATCH_SIZE]
            
            for j in range(0, n, self.BATCH_SIZE):
                dst_chunk = self.nodes[j : j + self.BATCH_SIZE]
                
                print(f"    - Processing Chunk: Rows {i}-{i+len(src_chunk)} | Cols {j}-{j+len(dst_chunk)}...", end="\r")
                
                chunk_matrix = self._fetch_chunk_with_retry(src_chunk, dst_chunk)
                
                if chunk_matrix is not None:
                    full_matrix[i : i + len(src_chunk), j : j + len(dst_chunk)] = chunk_matrix
                else:
                    print(f"\n    [CRITICAL] Chunk fail at {i}:{j}. Filling with Haversine fallback.")
                    fallback_matrix = self._calculate_haversine_subset(src_chunk, dst_chunk)
                    full_matrix[i : i + len(src_chunk), j : j + len(dst_chunk)] = fallback_matrix
        
        print("\n  > Matrix download complete.")
        full_matrix = np.nan_to_num(full_matrix, nan=1e9)
        return full_matrix

    #============================================================
    # MOST IMPORTANT, TO EXPORT DF_DIST_MATRIX AND DF_TIME_MATRIX
    #============================================================
    def calculate_matrices(self, avg_speed_kmh=30):
        # REUSE IF CACHE EXIST TO AVOID RECALCULATION
        if self._cached_distance_array_meters is None:
            self._cached_distance_array_meters = self._fetch_full_matrix_batched()
        else:
            print("  > Using cached raw distances...")

        dist_array = self._cached_distance_array_meters
        
        # SPEED
        speed = max(avg_speed_kmh, 1) 
        time_array = (dist_array / 1000.0) / speed * 60.0
        
        node_ids = [n['ID'] for n in self.nodes]
        df_dist = pd.DataFrame(dist_array, index=node_ids, columns=node_ids)
        df_time = pd.DataFrame(time_array, index=node_ids, columns=node_ids)
        
        return df_dist, df_time

    def export_demo_map(self, output_file='real_route.html'):
        if not self.nodes: return
        depot = self.nodes[0]
        m = folium.Map(location=[depot['Lat'], depot['Long']], zoom_start=12, tiles='CartoDB positron')
        bounds = []
        for node in self.nodes:
            loc = [node['Lat'], node['Long']]
            bounds.append(loc)
            if node['Type'] == 'Depot':
                folium.Marker(loc, popup=f"Depot: {node['ID']}", icon=folium.Icon(color='red', icon='home')).add_to(m)
            else:
                folium.CircleMarker(loc, radius=3, color='blue', fill=True, fill_opacity=0.7).add_to(m)
        
        if len(self.nodes) > 1:
            demo_points = self.nodes[:] + [self.nodes[0]]
            coords_str = ";".join([f"{n['Long']},{n['Lat']}" for n in demo_points])
            url = f"{self.osrm_route_url}{coords_str}?overview=full&geometries=polyline"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    decoded = polyline.decode(r.json()['routes'][0]['geometry'])
                    folium.PolyLine(decoded, color="green", weight=2.5, opacity=0.8).add_to(m)
            except: pass
            
        m.fit_bounds(bounds)
        m.save(output_file)
        print(f"  > Map exported: {output_file}")

if __name__ == "__main__":
    INPUT_FILE = 'Split_TransportOrder_1day.csv'
    DEPOT = INPUT_FILE[-8:-4]
    # Đặt tên folder output ở đây
    OUTPUT_DIR = "DistTimeMatrix"
    DRAW_MAP = True
    avg_speed_kmh = 30
    
    if os.path.exists(INPUT_FILE):
        print(">>>DISTANCE MATRIX GENERATOR<<<")
        
        calculator = DistanceMatrixCalculator(INPUT_FILE)
        
        # Bắt đầu tính toán
        df_dist, df_time = calculator.calculate_matrices(avg_speed_kmh=avg_speed_kmh)
        
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        df_dist.to_csv(f'{OUTPUT_DIR}/distance_matrix_{DEPOT}.csv')
        df_time.to_csv(f'{OUTPUT_DIR}/time_matrix_truck_{avg_speed_kmh}kmh_{DEPOT}.csv')
        if DRAW_MAP:
            calculator.export_demo_map(f'{OUTPUT_DIR}/map_all_nodes_{DEPOT}.html')
        print(f"Processing complete. Matrix shape: {df_dist.shape}")
    else:
        print(f"File {INPUT_FILE} not found.")