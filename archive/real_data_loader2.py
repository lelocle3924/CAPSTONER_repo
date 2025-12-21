# file: core/real_data_loader.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
from core.data_structures import ProblemData, VehicleType

class RealDataLoader:
    def __init__(self):
        # Định nghĩa Fleet mặc định (cấu hình chuẩn)
        self.default_fleet = [
            VehicleType(0, "MC",  150,  0.12, 40, 10,  5, 1, 50), #index, name, KGM, CBM, speed, 
            VehicleType(1, "AUV", 1000, 4.5,  30, 10, 10, 2, 20),
            VehicleType(2, "4w",  2000, 10.2, 30, 15, 12, 3, 15),
            VehicleType(3, "6w",  7000, 21.3, 30, 20, 14, 5, 10),
            VehicleType(4, "10w", 15000, 55.2, 30, 25, 18, 6, 5),
            VehicleType(5, "40ft", 28000, 69.0, 30, 35, 20, 9, 2)
        ]
        self.vehicle_map = {v.name: v.type_id for v in self.default_fleet}

    def _normalize_id(self, val) -> str:
        """
        Chuẩn hóa ID về dạng string số nguyên, loại bỏ .0
        """
        if pd.isna(val): return ""
        s = str(val).strip()
        if s.endswith('.0'):
            s = s[:-2]
        return s

    def _parse_time(self, time_str):
        """Chuyển đổi chuỗi giờ (HH:MM) sang phút tính từ 00:00"""
        if pd.isna(time_str): return 0
        try:
            if isinstance(time_str, str):
                parts = time_str.split(':')
                return int(parts[0]) * 60 + int(parts[1])
        except:
            pass
        return 0

    def _parse_allowed_trucks(self, allowed_str: str) -> List[int]:
        """Parse chuỗi '{AUV, 4w}' thành list ID"""
        if pd.isna(allowed_str): return [v.type_id for v in self.default_fleet]
        
        clean_str = str(allowed_str).replace('{', '').replace('}', '').replace(' ', '').replace('"', '')
        types = clean_str.split(',')
        
        allowed_ids = []
        for t in types:
            if t in self.vehicle_map:
                allowed_ids.append(self.vehicle_map[t])
        
        return allowed_ids if allowed_ids else [v.type_id for v in self.default_fleet]

    def load_day_data(self, order_csv: str, dist_csv: str, time_csv: str) -> ProblemData:
        print(f"--- Loading Data from {order_csv} ---")
        
        # 1. Load DataFrames
        df_orders_raw = pd.read_csv(order_csv)
        
        # --- NEW: CHECK MULTIPLE DAYS ---
        if 'DeliveryDate' in df_orders_raw.columns:
            unique_dates = df_orders_raw['DeliveryDate'].nunique()
            if unique_dates > 1:
                print(f"[DATA LOADER]: More than 1 day detected among orders ({unique_dates} dates found).")
                print("               Proceeding to aggregate all, but results may be mixed.")
        # --------------------------------

        # --- NEW: AGGREGATE CUSTOMERS ---
        # Gộp các đơn hàng của cùng 1 Customer ID
        print(f"  > Raw Orders: {len(df_orders_raw)}")
        
        # Xử lý NaN trước khi sum để tránh lỗi
        df_orders_raw['KGM'] = df_orders_raw['KGM'].fillna(0)
        df_orders_raw['CBM'] = df_orders_raw['CBM'].fillna(0)
        
        # Định nghĩa cách gộp
        agg_rules = {
            'KGM': 'sum',
            'CBM': 'sum',
            'CusLat': 'first',      # Giả định cùng ID thì cùng vị trí
            'CusLong': 'first',
            'Beginning1': 'first',  # Giả định cùng ID thì cùng Time Window
            'Ending1': 'first',
            'DwellTime': 'first',   # Lấy dwell time của dòng đầu (hoặc max nếu muốn an toàn)
            'AllowedTrucks': 'first',
            'Depot': 'first',       # Thông tin Depot giữ nguyên
            'DepotLat': 'first',
            'DepotLong': 'first'
        }
        
        # Thực hiện GroupBy
        df_orders = df_orders_raw.groupby('Customer', as_index=False).agg(agg_rules)
        print(f"  > Unique Customers (Stops) after aggregation: {len(df_orders)}")
        # --------------------------------

        # Load matrix
        df_dist = pd.read_csv(dist_csv, index_col=0, dtype=str) 
        df_time = pd.read_csv(time_csv, index_col=0, dtype=str)
        df_dist = df_dist.astype(float)
        df_time = df_time.astype(float)

        # 2. Chuẩn hóa ID của Matrix
        df_dist.index = df_dist.index.map(self._normalize_id)
        df_dist.columns = df_dist.columns.map(self._normalize_id)
        df_time.index = df_time.index.map(self._normalize_id)
        df_time.columns = df_time.columns.map(self._normalize_id)

        # 3. Xử lý Depot ID
        raw_depot_id = df_orders.iloc[0]['Depot']
        depot_id = self._normalize_id(raw_depot_id)
        
        if 'Depot' in df_dist.index and depot_id != 'Depot':
            # print(f"  > Mapping Matrix 'Depot' -> Order ID '{depot_id}'")
            df_dist.rename(index={'Depot': depot_id}, columns={'Depot': depot_id}, inplace=True)
            df_time.rename(index={'Depot': depot_id}, columns={'Depot': depot_id}, inplace=True)

        # 4. Tạo danh sách node_ids
        node_ids = [depot_id] 
        cust_ids = df_orders['Customer'].map(self._normalize_id).tolist()
        node_ids.extend(cust_ids)
        
        num_nodes = len(node_ids)

        # 5. Khởi tạo mảng dữ liệu
        coords = np.zeros((num_nodes, 2))
        demands_kg = np.zeros(num_nodes)
        demands_cbm = np.zeros(num_nodes)
        time_windows = np.zeros((num_nodes, 2))
        service_times = np.zeros(num_nodes)
        allowed_vehicles = []

        # -- Fill Depot Data (Index 0) --
        depot_row = df_orders.iloc[0]
        coords[0] = [depot_row['DepotLat'], depot_row['DepotLong']]
        time_windows[0] = [0, 24 * 60] 
        allowed_vehicles.append([v.type_id for v in self.default_fleet])

        # -- Fill Customer Data (Index 1..N) --
        # Loop qua df_orders đã được Aggregated
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

        # 6. Reindex Matrix
        try:
            missing_ids = [nid for nid in set(node_ids) if nid not in df_dist.index]
            if missing_ids:
                print(f"  [WARNING] Found {len(missing_ids)} IDs in Order but missing in Matrix.")
            
            final_dist = df_dist.reindex(index=node_ids, columns=node_ids, fill_value=1e9).to_numpy()
            final_time = df_time.reindex(index=node_ids, columns=node_ids, fill_value=1e9).to_numpy()
            
            np.fill_diagonal(final_dist, 0)
            np.fill_diagonal(final_time, 0)
            
        except Exception as e:
            print(f"Matrix Reindexing Error: {e}")
            raise e

        print("--- Data Loading Complete ---\n")
        
        return ProblemData(
            dist_matrix=final_dist,
            time_matrix=final_time,
            node_ids=node_ids,
            coords=coords,
            demands_kg=demands_kg,
            demands_cbm=demands_cbm,
            time_windows=time_windows,
            service_times=service_times,
            allowed_vehicles=allowed_vehicles,
            vehicle_types=self.default_fleet
        )