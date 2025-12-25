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
        if pd.isna(allowed_str): return [v.type_id for v in self.fleet]
        
        clean_str = str(allowed_str).replace('{', '').replace('}', '').replace(' ', '').replace('"', '')
        types = clean_str.split(',')
        
        allowed_ids = []
        for t in types:
            if t in self.vehicle_map:
                allowed_ids.append(self.vehicle_map[t])
        
        return allowed_ids if allowed_ids else [v.type_id for v in self.fleet]

    def _load_fleet_from_csv(self, truck_csv_path: str):
        """
        Đọc TruckMaster.csv để tạo VehicleType objects (dùng cho ProblemData)
        """
        print(f"  > Loading Fleet Config from {truck_csv_path}...")
        try:
            df_truck = pd.read_csv(truck_csv_path)
            self.fleet = []
            
            for idx, row in df_truck.iterrows():
                v = VehicleType(
                    type_id=idx,
                    name=str(row['TruckName']).strip(),
                    capacity_kg=float(row['CapacityKg']),
                    capacity_cbm=float(row['CapacityCbm']),
                    speed_kmh=float(row['AverageSpeedKmH']),
                    fixed_cost=float(row['FixedCost']),
                    cost_per_km=float(row['CostPerKm']),
                    cost_per_hour=float(row['CostPerHour']),
                    count=50 
                )
                self.fleet.append(v)
            
            self.vehicle_map = {v.name: v.type_id for v in self.fleet}
            print(f"  > Fleet Loaded: {len(self.fleet)} vehicle types.")
            
        except Exception as e:
            print(f"[CRITICAL] Error loading Truck Master: {e}")
            raise e

    def load_day_data(self, order_csv_path: str, truck_csv_path: str) -> ProblemData:
        """
        Load orders, Load Fleet, and receive Matrices from DistanceMatrixCalculator.
        """
        print(f"--- Loading Data Pipeline ---")
        
        # 1. LOAD FLEET Objects (để lấy Attributes: cost, weight, volume)
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

        # 3. GET MATRICES FROM CALCULATOR
        print("  > Invoking DistanceMatrixCalculator...")
        # Truyền cả 2 file path vào đây
        calculator = DistanceMatrixCalculator(order_csv_path, truck_csv_path)
        
        # Hàm này giờ trả về numpy array trực tiếp
        dist_matrix_meters, super_time_matrix = calculator.calculate_matrices() 
        
        # Sửa lại diagonal cho chắc chắn
        np.fill_diagonal(dist_matrix_meters, 0)
            
        # 4. Fill Node Attributes
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