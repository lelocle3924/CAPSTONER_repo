import pandas as pd
import os

#ver 2
class TransportOrderSplitter:
    def __init__(self, input_file_path, output_dir):
        self.input_file_path = input_file_path
        self.df = None
        self.output_dir = output_dir
        # Mapping đổi tên cột
        self.column_mapping = {
            'Shipping Point': 'Depot',
            'DepotLongitude': 'DepotLong',
            'DepotLatitude': 'DepotLat',
            'Ship to': 'Customer',
            'CusLongitude': 'CusLong',
            'CusLatitude': 'CusLat',
            'Planned Delivery Date': 'DeliveryDate',
            'Number of HUs': 'HUs',
            'Dwell Time': 'DwellTime',
            'Total Volume': 'CBM',
            'Total Weight': 'KGM'
            # Các cột DOW, Beginning1... giữ nguyên tên
        }

    def load_data(self):
        """Đọc file Aggregate"""
        if not os.path.exists(self.input_file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {self.input_file_path}")
        print(f">>> Đang đọc file {self.input_file_path}...")
        self.df = pd.read_csv(self.input_file_path)

    def rename_columns(self):
        """Đổi tên cột theo yêu cầu"""
        print(">>> Đang đổi tên cột...")
        self.df.rename(columns=self.column_mapping, inplace=True)
        
        # --- CẬP NHẬT MỚI: Đảm bảo cột DeliveryDate là dạng ngày tháng để sort đúng ---
        if 'DeliveryDate' in self.df.columns:
            self.df['DeliveryDate'] = pd.to_datetime(self.df['DeliveryDate'])

    def split_and_save(self):
        if 'Depot' not in self.df.columns:
            raise ValueError("Không tìm thấy cột 'Depot' (Shipping Point cũ) để chia tách.")

        unique_depots = self.df['Depot'].unique()
        print(f">>> Tìm thấy {len(unique_depots)} Depot: {unique_depots}")

        for depot in unique_depots:
            # Lọc dữ liệu cho từng Depot
            # Dùng .copy() để tránh cảnh báo SettingWithCopyWarning của Pandas
            sub_df = self.df[self.df['Depot'] == depot].copy()
            
            # --- CẬP NHẬT MỚI: Sắp xếp theo ngày giao hàng (Tăng dần) ---
            sub_df = sub_df.sort_values(by='DeliveryDate', ascending=True)
            
            # Tạo tên file output
            safe_depot_name = str(depot).replace('/', '_').replace('\\', '_')
            output_name = f"{self.output_dir}/Split_TransportOrder_{safe_depot_name}.csv"
            
            # Xuất file (giữ định dạng ngày tháng chuẩn YYYY-MM-DD)
            sub_df.to_csv(output_name, index=False)
            print(f"    -> Đã xuất file: {output_name} ({len(sub_df)} dòng)")

    def run(self):
        self.load_data()
        self.rename_columns()
        self.split_and_save()

# --- PHẦN THỰC THI CODE ---
if __name__ == "__main__":
    # Đường dẫn đến file Aggregate (File SẠCH tạo ra từ bước trước)
    DIR = "CleanData"
    input_aggregate_file = f"{DIR}/AggregateTransportOrder.csv"
    
    splitter = TransportOrderSplitter(input_aggregate_file, output_dir=DIR)
    splitter.run()