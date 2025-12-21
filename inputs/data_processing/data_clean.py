import pandas as pd
import numpy as np
import os
from datetime import datetime

#ver 7
#fix unit KGM và CBM, remove duplicates, handle exceptional CBM, KGM

class TransportOrderProcessor:
    def __init__(self, order_files, customer_master_path, truck_master_path, depot_master_path):
        self.order_files = order_files
        self.customer_master_path = customer_master_path
        self.truck_master_path = truck_master_path
        self.depot_master_path = depot_master_path
        self.df_combined = None
        self.df_customer = None
        self.df_truck = None
        self.df_depot = None

    def _clean_currency_string(self, x):
        """Chuyển đổi chuỗi số có dấu phẩy thành float."""
        if isinstance(x, str):
            clean_str = x.replace(',', '').strip()
            try:
                return float(clean_str)
            except ValueError:
                return 0.0
        return x

    def _convert_to_24h(self, time_str):
        """Chuyển đổi chuỗi giờ AM/PM sang format 24h (HH:MM)."""
        if pd.isna(time_str) or str(time_str).strip() == '':
            return None
        try:
            time_obj = pd.to_datetime(time_str.strip()).time()
            return time_obj.strftime('%H:%M')
        except:
            return None

    def load_and_merge_orders(self):
        """Đọc file, xử lý đơn vị (Unit Conversion) và loại bỏ trùng lặp."""
        print(">>> Đang load và gộp dữ liệu Transport Order...")
        list_df = []
        for file_path in self.order_files:
            try:
                df = pd.read_csv(file_path)
                list_df.append(df)
            except Exception as e:
                print(f"Lỗi khi đọc file {file_path}: {e}")

        if not list_df:
            raise ValueError("Không đọc được file order nào.")

        self.df_combined = pd.concat(list_df, ignore_index=True)

        # 1. Xử lý các cột số (Xóa dấu phẩy)
        cols_to_fix = ['Total Weight', 'Total Volume', 'Number of HUs']
        for col in cols_to_fix:
            if col in self.df_combined.columns:
                self.df_combined[col] = self.df_combined[col].apply(self._clean_currency_string)
            else:
                self.df_combined[col] = 0.0

        # --- FIX 1: UNIT CONVERSION (QUAN TRỌNG) ---
        print(">>> Đang chuyển đổi đơn vị (Gram -> KG, CCM -> CBM)...")
        # Gram -> KG
        self.df_combined['Total Weight'] = self.df_combined['Total Weight'] / 1000.0
        # CCM -> CBM
        self.df_combined['Total Volume'] = self.df_combined['Total Volume'] / 1_000_000.0

        # --- FIX 2: REMOVE DUPLICATES (QUAN TRỌNG) ---
        # Loại bỏ các dòng trùng hoàn toàn
        before_dedup = len(self.df_combined)
        self.df_combined.drop_duplicates(inplace=True)
        after_dedup = len(self.df_combined)
        print(f">>> Đã loại bỏ {before_dedup - after_dedup} dòng trùng lặp.")

        # Xử lý ngày tháng
        self.df_combined['Planned Delivery Date'] = pd.to_datetime(self.df_combined['Planned Delivery Date'])
        self.df_combined['DOW'] = self.df_combined['Planned Delivery Date'].dt.day_name()
        
        # Chuẩn hóa Shipping Point
        if 'Shipping Point' in self.df_combined.columns:
            self.df_combined['Shipping Point'] = self.df_combined['Shipping Point'].astype(str)
            self.df_combined['Shipping Point'] = self.df_combined['Shipping Point'].apply(lambda x: x.split('.')[0])
        
        print(f"Tổng số đơn hàng sau khi làm sạch: {len(self.df_combined)}")

    def process_depot_info(self):
        """Lấy tọa độ Depot."""
        print(">>> Đang xử lý thông tin Depot...")
        self.df_depot = pd.read_csv(self.depot_master_path)
        self.df_depot['DepotID'] = self.df_depot['DepotID'].astype(str)
        
        self.df_combined = self.df_combined.merge(
            self.df_depot[['DepotID', 'Latitude', 'Longitude']],
            left_on='Shipping Point',
            right_on='DepotID',
            how='left'
        )
        
        self.df_combined.rename(columns={
            'Longitude': 'DepotLongitude', 
            'Latitude': 'DepotLatitude'
        }, inplace=True)
        
        if 'DepotID' in self.df_combined.columns:
            self.df_combined.drop(columns=['DepotID'], inplace=True)

    def process_customer_info(self, handling_time_per_hu):
        """Merge Customer Master và tính toán Dwell Time."""
        print(">>> Đang xử lý thông tin Customer...")
        self.df_customer = pd.read_csv(self.customer_master_path, low_memory=False)
        
        self.df_customer['ShipToRef'] = self.df_customer['ShipToRef'].astype(str)
        if 'Ship to' in self.df_combined.columns:
            self.df_combined['Ship to'] = self.df_combined['Ship to'].astype(str)

        # Merge
        cols_to_use = ['ShipToRef', 'Latitude', 'Longitude', 'DwellTimeInHour',
                       'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        if 'DwellTimeInHour' not in self.df_customer.columns:
             self.df_customer['DwellTimeInHour'] = np.nan

        self.df_combined = self.df_combined.merge(
            self.df_customer[cols_to_use],
            left_on='Ship to',
            right_on='ShipToRef',
            how='left'
        )

        self.df_combined.rename(columns={'Longitude': 'CusLongitude', 'Latitude': 'CusLatitude'}, inplace=True)

        # Time Windows
        def extract_time_window_complex(row):
            dow = row['DOW']
            if pd.isna(dow) or dow not in row:
                return None, None, None, None
            
            raw_window = row[dow]
            if pd.isna(raw_window) or not isinstance(raw_window, str):
                return None, None, None, None

            b1, e1, b2, e2 = None, None, None, None
            sessions = raw_window.split('/')
            
            if len(sessions) > 0 and '-' in sessions[0]:
                parts1 = sessions[0].split('-')
                b1 = self._convert_to_24h(parts1[0])
                e1 = self._convert_to_24h(parts1[1])
            
            if len(sessions) > 1 and '-' in sessions[1]:
                parts2 = sessions[1].split('-')
                b2 = self._convert_to_24h(parts2[0])
                e2 = self._convert_to_24h(parts2[1])
            return b1, e1, b2, e2

        time_result = self.df_combined.apply(extract_time_window_complex, axis=1, result_type='expand')
        self.df_combined[['Beginning1', 'Ending1', 'Beginning2', 'Ending2']] = time_result

        # Dwell Time
        def _calc_dwell_time(row):
            if pd.notna(row['DwellTimeInHour']) and row['DwellTimeInHour'] > 0:
                return row['DwellTimeInHour']
            
            hus = row['Number of HUs'] if pd.notna(row['Number of HUs']) else 0
            # Dwell Time = Fixed Setup (0.25h) + Variable (HU * time/HU)
            return 0.25 + (handling_time_per_hu * hus)

        self.df_combined['Dwell Time'] = self.df_combined.apply(_calc_dwell_time, axis=1)

    def process_truck_info(self):
        """Merge ShipToTruckMaster."""
        print(">>> Đang xử lý thông tin Truck Master...")
        self.df_truck = pd.read_csv(self.truck_master_path)
        self.df_truck['ShipToRef'] = self.df_truck['ShipToRef'].astype(str)

        self.df_combined = self.df_combined.merge(
            self.df_truck[['ShipToRef', 'TruckAllowed']],
            left_on='Ship to',
            right_on='ShipToRef',
            how='left'
        )
        self.df_combined.rename(columns={'TruckAllowed': 'AllowedTrucks'}, inplace=True)

    def finalize_and_export(self, valid_output_path, exception_output_path):
        """Lọc dữ liệu lỗi (Tọa độ thiếu hoặc Weight/Volume quá khổ) và xuất file."""
        print(">>> Đang hoàn thiện và kiểm tra Extreme Outliers...")
        
        target_columns = [
            'Shipping Point', 'DepotLongitude', 'DepotLatitude', 'Ship to', 
            'CusLongitude', 'CusLatitude', 'Planned Delivery Date', 'DOW', 
            'Beginning1', 'Ending1', 'Beginning2', 'Ending2',
            'Number of HUs', 'Dwell Time', 'Total Volume', 'Total Weight', 'AllowedTrucks'
        ]

        for col in target_columns:
            if col not in self.df_combined.columns:
                self.df_combined[col] = np.nan

        df_final = self.df_combined[target_columns]

        # --- FIX 3: OUTLIER FILTERING (DỮ LIỆU ẢO MA) ---
        # Ngưỡng (Threshold): 
        # Weight > 25,000 KG (25 tấn - quá tải xe thường)
        # Volume > 60 CBM (quá to so với xe tải thường)
        # Hoặc thiếu tọa độ
        
        mask_coords = df_final['CusLongitude'].notna() & df_final['CusLatitude'].notna()
        mask_weight = df_final['Total Weight'] <= 25000 
        mask_volume = df_final['Total Volume'] <= 60

        # Đơn hợp lệ phải thỏa mãn TẤT CẢ điều kiện
        mask_valid = mask_coords & mask_weight & mask_volume
        
        df_valid = df_final[mask_valid]
        df_exceptions = df_final[~mask_valid]

        print("-" * 50)
        print(f"TỔNG KẾT DỮ LIỆU:")
        print(f"  > Tổng số dòng: {len(df_final)}")
        print(f"  > Đơn HỢP LỆ (Feasible): {len(df_valid)}")
        print(f"  > Đơn LỖI (Exceptions): {len(df_exceptions)}")
        if not df_exceptions.empty:
            print("    (Bao gồm: Thiếu tọa độ, Weight > 25 Tấn, hoặc Volume > 60 CBM)")
        print("-" * 50)

        df_valid.to_csv(valid_output_path, index=False)
        print(f">>> File SẠCH đã lưu tại: {valid_output_path}")

        if not df_exceptions.empty:
            df_exceptions.to_csv(exception_output_path, index=False)
            print(f">>> File NGOẠI LỆ đã lưu tại: {exception_output_path}")

    def run(self, handling_time_per_hu, output_filename, exception_filename):
        self.load_and_merge_orders()
        self.process_depot_info()
        self.process_customer_info(handling_time_per_hu)
        self.process_truck_info()
        self.finalize_and_export(output_filename, exception_filename)

# --- EXECUTION ---
if __name__ == "__main__":
    # --- PATHS ---
    path_cebu = "TransportData/Cebu_TransportOrder.csv" 
    path_starosa = "TransportData/StaRosa_TransportOrder.csv"
    path_taguig = "TransportData/Taguig_TransportOrder.csv"
    path_tdc = "TransportData/TDC_TransportOrder.csv"
    path_canlubang = "TransportData/Canlubang_TransportOrder.csv"
    path_sg = "TransportData/SG_TransportOrder.csv"
    
    path_customer_master = "MasterData/CustomerMaster.csv"
    path_truck_master = "MasterData/ShipToTruckMaster.csv"
    path_depot_master = "MasterData/DepotMaster.csv"
    
    OUTPUT_DIR = "CleanData"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/AggregateTransportOrder.csv"
    exception_file = f"{OUTPUT_DIR}/ExceptionCases.csv"

    # handling_time_per_HU (đơn vị: Giờ)
    # Ví dụ: 6 phút/HU = 0.1 giờ. ---> Thời gian xử lý 1 đơn vị hàng (vdu 1 thùng)
    handling_time_per_HU = 0.1

    order_files_list = [path_cebu, path_starosa, path_taguig, path_tdc, path_sg, path_canlubang]

    processor = TransportOrderProcessor(
        order_files=order_files_list,
        customer_master_path=path_customer_master,
        truck_master_path=path_truck_master,
        depot_master_path=path_depot_master
    )
    
    processor.run(
        handling_time_per_hu=handling_time_per_HU,
        output_filename=output_file,
        exception_filename=exception_file
    )