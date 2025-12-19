import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from data_structures import ProblemData, VehicleType

class RealDataLoader:
    def __init__(self):
        # Định nghĩa Fleet mặc định (dựa trên Case Study)
        # Em tạm fix cứng ở đây, sau này sẽ load từ file Config hoặc CSV VehicleMaster
        self.default_fleet = [
            VehicleType(0, "MC",  100,  0.12, 30, 50,  10, 50),
            VehicleType(1, "AUV", 800,  2.5,  30, 500, 25, 20),
            VehicleType(2, "4w",  1500, 5.0,  30, 1000, 40, 15),
            VehicleType(3, "6w",  3000, 12.0, 25, 2000, 60, 10),
            VehicleType(4, "10w", 12000, 30.0, 25, 3500, 80, 5),
            VehicleType(5, "40ft", 28000, 60.0, 20, 5000, 100, 2)
        ]
        self.vehicle_map = {v.name: v.type_id for v in self.default_fleet}

    def _parse_time(self, time_str, base_date_str="2023-01-01"):
        """Chuyển đổi chuỗi giờ (HH:MM) sang phút tính từ 00:00"""
        if pd.isna(time_str): return 0
        try:
            # Giả sử format HH:MM
            t = datetime.strptime(time_str, "%H:%M")
            return t.hour * 60 + t.minute
        except:
            return 0

    def _parse_allowed_trucks(self, allowed_str: str) -> List[int]:
        """Parse chuỗi '{AUV, MC, 4w}' thành list ID [1, 0, 2]"""
        if pd.isna(allowed_str): return [v.type_id for v in self.default_fleet] # Mặc định cho phép tất cả
        
        # Clean string: bỏ dấu {}, split dấu phẩy
        clean_str = allowed_str.replace('{', '').replace('}', '').replace(' ', '')
        types = clean_str.split(',')
        
        allowed_ids = []
        for t in types:
            if t in self.vehicle_map:
                allowed_ids.append(self.vehicle_map[t])
        
        # Nếu không map được gì (dữ liệu lỗi), cho phép tất cả để tránh crash
        return allowed_ids if allowed_ids else [v.type_id for v in self.default_fleet]

    def load_day_data(self, order_csv: str, dist_csv: str, time_csv: str) -> ProblemData:
        # 1. Load DataFrames
        df_orders = pd.read_csv(order_csv)
        df_dist = pd.read_csv(dist_csv, index_col=0)
        df_time = pd.read_csv(time_csv, index_col=0)

        # 2. Xử lý Depot & Customers
        # Depot là dòng đầu tiên trong split file
        depot_row = df_orders.iloc[0]
        
        # Tạo list node_ids chuẩn: [DepotID, Customer1, Customer2...]
        # Lưu ý: df_dist và df_time headers phải khớp với IDs này
        node_ids = [str(depot_row['Depot'])] # Index 0
        customers = df_orders['Customer'].astype(str).tolist()
        node_ids.extend(customers)
        
        num_nodes = len(node_ids)

        # 3. Khởi tạo mảng dữ liệu
        coords = np.zeros((num_nodes, 2))
        demands_kg = np.zeros(num_nodes)
        demands_cbm = np.zeros(num_nodes)
        time_windows = np.zeros((num_nodes, 2))
        service_times = np.zeros(num_nodes)
        allowed_vehicles = []

        # -- Fill Depot Data (Index 0) --
        coords[0] = [depot_row['DepotLat'], depot_row['DepotLong']]
        demands_kg[0] = 0
        demands_cbm[0] = 0
        # Depot time window: Mở cửa 8h sáng -> 5h chiều (hoặc rộng hơn tùy logic)
        # Tạm lấy theo Opening/Closing của Customer đầu tiên làm chuẩn hoặc fix cứng
        # Ở đây em fix cứng Depot mở 24/24 hoặc theo ca làm việc (0 -> 1440 phút)
        time_windows[0] = [0, 1440] 
        service_times[0] = 0
        allowed_vehicles.append([v.type_id for v in self.default_fleet]) # Depot cho phép mọi xe

        # -- Fill Customer Data (Index 1..N) --
        base_start_time = 8 * 60 # 8:00 AM làm mốc 0 nếu muốn, hoặc dùng tuyệt đối. 
        # Để đơn giản, em dùng phút trong ngày (00:00 = 0)
        
        for i, row in df_orders.iterrows():
            idx = i + 1 # Index trong mảng (0 là depot)
            
            coords[idx] = [row['CusLat'], row['CusLong']]
            demands_kg[idx] = row['KGM']
            demands_cbm[idx] = row['CBM']
            
            # Time Windows
            start_min = self._parse_time(row['Beginning1'])
            end_min = self._parse_time(row['Ending1'])
            if end_min < start_min: end_min = 1440 # Fix lỗi data nếu có
            time_windows[idx] = [start_min, end_min]
            
            # Service Time (DwellTime đang tính bằng giờ -> convert sang phút)
            service_times[idx] = row['DwellTime'] * 60 
            
            # Allowed Trucks
            allowed_vehicles.append(self._parse_allowed_trucks(row['AllowedTrucks']))

        # 4. Re-index Distance/Time Matrices theo đúng thứ tự node_ids
        # Đảm bảo ma trận vuông và đúng thứ tự [Depot, C1, C2...]
        final_dist = np.zeros((num_nodes, num_nodes))
        final_time = np.zeros((num_nodes, num_nodes))
        
        # Check xem các ID trong order có tồn tại trong ma trận khoảng cách không
        available_ids_dist = set(df_dist.index.astype(str))
        
        # Fallback: Nếu thiếu ID trong matrix, dùng Euclidean (tạm thời)
        # Nhưng ở đây ta assume Nhóm A đã làm matrix chuẩn theo file order
        
        # Dùng numpy indexing hoặc loc của pandas (loc chậm hơn nhưng an toàn cho mapping)
        # Để tối ưu, ta filter df_dist trước
        # Cần xử lý kỹ vụ ID là float hay int trong CSV string
        
        # Mapping index matrix A -> index array B
        # (Em sẽ viết code simplified đoạn này, thực tế cần try-catch kỹ)
        try:
            # Lấy sub-dataframe đúng thứ tự node_ids
            # Lưu ý: node_ids trong file csv có thể là '2524' nhưng trong matrix là 2524.0
            # Cần normalize ID
            matrix_cols = df_dist.columns.astype(str).str.replace('.0', '', regex=False)
            df_dist.columns = matrix_cols
            df_dist.index = df_dist.index.astype(str).str.replace('.0', '', regex=False)
            
            df_time.columns = matrix_cols
            df_time.index = df_dist.index # Assume time index same as dist index

            # Reindex
            # Fillna bằng giá trị lớn vô cùng nếu không tìm thấy đường
            final_dist = df_dist.reindex(index=node_ids, columns=node_ids, fill_value=1e9).to_numpy()
            final_time = df_time.reindex(index=node_ids, columns=node_ids, fill_value=1e9).to_numpy()
            
        except Exception as e:
            print(f"Matrix Mapping Error: {e}. Checking logic...")
            # Fallback logic sẽ code sau nếu cần thiết
            
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