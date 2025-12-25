# core/real_data_loader.py 
# ver 3: super time matrix, gọi DistanceMatrixCalculator
# ver 3.5: có thêm read fleet_csv_path

import pandas as pd
import numpy as np
from typing import List
from core.data_structures import ProblemData, VehicleType
from core.distance_matrix import DistanceMatrixCalculator 

class RealDataLoader:
    def __init__(self):
        self.fleet: List[VehicleType] = []
        self.vehicle_map = {}

    def _normalize_id(self, val) -> str:
        if pd.isna(val): return ""
        s = str(val).strip()
        if s.endswith('.0'): s = s[:-2]
        return s

    def _parse_time(self, time_str):
        if pd.isna(time_str): return 0
        try:
            if isinstance(time_str, str):
                parts = time_str.split(':')
                return int(parts[0]) * 60 + int(parts[1])
        except: pass
        return 0

    def _parse_allowed_trucks(self, allowed_str: str) -> List[int]:
        """
        Parse allowed trucks string. 
        Nếu không quy định (NaN), mặc định cho phép tất cả các loại xe trong Fleet.
        """
        if pd.isna(allowed_str): return [v.type_id for v in self.fleet]
        
        clean_str = str(allowed_str).replace('{', '').replace('}', '').replace(' ', '').replace('"', '')
        types = clean_str.split(',')
        
        allowed_ids = []
        for t in types:
            if t in self.vehicle_map:
                allowed_ids.append(self.vehicle_map[t])
        
        # Nếu parse ra rỗng (do tên sai hoặc format lạ), fallback về all allowed
        return allowed_ids if allowed_ids else [v.type_id for v in self.fleet]

    def _load_fleet_from_csv(self, truck_csv_path: str):
        """
        Đọc TruckMaster.csv và khởi tạo self.fleet
        """
        print(f"  > Loading Fleet Config from {truck_csv_path}...")
        try:
            df_truck = pd.read_csv(truck_csv_path)
            self.fleet = []
            
            for idx, row in df_truck.iterrows():
                # Tự động gán type_id dựa trên index dòng
                v = VehicleType(
                    type_id=idx,
                    name=str(row['TruckName']).strip(),
                    capacity_kg=float(row['CapacityKg']),
                    capacity_cbm=float(row['CapacityCbm']),
                    speed_kmh=float(row['AverageSpeedKmH']),
                    fixed_cost=float(row['FixedCost']),
                    cost_per_km=float(row['CostPerKm']),
                    cost_per_hour=float(row['CostPerHour']),
                    count=50 # Default availability (Infinite/Large number)
                )
                self.fleet.append(v)
            
            # Update map name -> id để dùng cho việc parse AllowedTrucks
            self.vehicle_map = {v.name: v.type_id for v in self.fleet}
            print(f"  > Fleet Loaded: {len(self.fleet)} vehicle types.")
            
        except Exception as e:
            print(f"[CRITICAL] Error loading Truck Master: {e}")
            raise e

    def load_day_data(self, order_csv_path: str, truck_csv_path: str) -> ProblemData:
        """
        Load orders, Load Fleet, and generate Matrices.
        """
        print(f"--- Loading Data Pipeline ---")
        
        # 1. LOAD FLEET FIRST (Để có thông tin speed tính matrix)
        self._load_fleet_from_csv(truck_csv_path)
        
        # 2. Load & Aggregate Orders
        print(f"  > Loading Orders from {order_csv_path}...")
        df_orders_raw = pd.read_csv(order_csv_path)
        
        df_orders_raw['KGM'] = df_orders_raw['KGM'].fillna(0)
        df_orders_raw['CBM'] = df_orders_raw['CBM'].fillna(0)
        
        agg_rules = {
            'KGM': 'sum', 'CBM': 'sum', 
            'CusLat': 'first', 'CusLong': 'first',
            'Beginning1': 'first', 'Ending1': 'first',
            'DwellTime': 'first', 'AllowedTrucks': 'first',
            'Depot': 'first', 'DepotLat': 'first', 'DepotLong': 'first'
        }
        df_orders = df_orders_raw.groupby('Customer', as_index=False).agg(agg_rules)
        
        raw_depot_id = df_orders.iloc[0]['Depot']
        depot_id = self._normalize_id(raw_depot_id)
        
        node_ids = [depot_id] + df_orders['Customer'].map(self._normalize_id).tolist()
        num_nodes = len(node_ids)

        # 3. GENERATE MATRICES
        print("  > Calculating Distance Matrix (OSRM)...")
        calculator = DistanceMatrixCalculator(order_csv_path)
        # Chỉ cần lấy distance matrix (meters), speed không quan trọng ở bước này
        df_dist_raw, _ = calculator.calculate_matrices(avg_speed_kmh=30) 
        
        if 'Depot' in df_dist_raw.index and depot_id != 'Depot':
             df_dist_raw.rename(index={'Depot': depot_id}, columns={'Depot': depot_id}, inplace=True)
             
        dist_matrix_meters = df_dist_raw.reindex(index=node_ids, columns=node_ids, fill_value=1e9).to_numpy()
        np.fill_diagonal(dist_matrix_meters, 0)

        # 4. BUILD SUPER TIME MATRIX (V, N, N)
        print("  > Building Super Time Matrix for Heterogeneous Fleet...")
        num_vehicle_types = len(self.fleet)
        super_time_matrix = np.zeros((num_vehicle_types, num_nodes, num_nodes))

        for v in self.fleet:
            # Convert km/h -> m/min
            speed_mpm = v.speed_kmh * 1000.0 / 60.0
            if speed_mpm <= 0.1:
                print(f"  Warning: Invalid speed for vehicle {v.name}: {speed_mpm} m/min, using safe speed 0.1 m/min")
                speed_mpm = 0.1
            
            # Time (min) = Distance (m) / Speed (m/min)
            time_matrix_v = dist_matrix_meters / speed_mpm
            super_time_matrix[v.type_id] = time_matrix_v
            
        # 5. Fill Attributes
        coords = np.zeros((num_nodes, 2))
        demands_kg = np.zeros(num_nodes)
        demands_cbm = np.zeros(num_nodes)
        time_windows = np.zeros((num_nodes, 2))
        service_times = np.zeros(num_nodes)
        allowed_vehicles = []

        # Depot
        depot_row = df_orders.iloc[0]
        coords[0] = [depot_row['DepotLat'], depot_row['DepotLong']]
        time_windows[0] = [0, 24 * 60] 
        allowed_vehicles.append([v.type_id for v in self.fleet])

        # Customers
        for i, row in df_orders.iterrows():
            idx = i + 1
            coords[idx] = [row['CusLat'], row['CusLong']]
            demands_kg[idx] = float(row['KGM'])
            demands_cbm[idx] = float(row['CBM'])
            
            start = self._parse_time(row['Beginning1'])
            end = self._parse_time(row['Ending1'])
            if end <= start: end = 18 * 60 
            time_windows[idx] = [start, end]
            
            dwell = float(row['DwellTime']) if not pd.isna(row['DwellTime']) else 0.5
            service_times[idx] = dwell * 60
            
            # Parse allowed trucks based on the loaded fleet map
            allowed_vehicles.append(self._parse_allowed_trucks(row['AllowedTrucks']))

        print("--- Data Loading Complete ---")
        return ProblemData(
            dist_matrix=dist_matrix_meters,
            super_time_matrix=super_time_matrix,
            node_ids=node_ids,
            coords=coords,
            demands_kg=demands_kg,
            demands_cbm=demands_cbm,
            time_windows=time_windows,
            service_times=service_times,
            allowed_vehicles=allowed_vehicles,
            vehicle_types=self.fleet
        )