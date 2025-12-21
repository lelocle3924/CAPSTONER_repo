import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import folium
import os
import numpy as np

class TransportVisualizer:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_distinct_colors(self, n):
        colormap = cm.get_cmap('tab10' if n <= 10 else 'gist_rainbow', n)
        colors = []
        for i in range(n):
            rgb = colormap(i)[:3]
            hex_color = mcolors.to_hex(rgb)
            colors.append(hex_color)
        return colors

    def plot_aggregate(self, file_path):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Lỗi đọc file: {e}")
            return

        df = df.dropna(subset=['CusLatitude', 'CusLongitude'])

        if df.empty:
            print("File has no valid coordinates")
            return

        depots = df['Shipping Point'].unique()
        colors = self._get_distinct_colors(len(depots))
        depot_color_map = dict(zip(depots, colors))

        #MATPLOTLIB PNG 
        plt.figure(figsize=(12, 10))
        
        for depot in depots:
            subset = df[df['Shipping Point'] == depot]
            plt.scatter(
                subset['CusLongitude'], 
                subset['CusLatitude'], 
                c=depot_color_map[depot], 
                label=f"Depot {depot}", 
                alpha=0.6, 
                edgecolors='w',
                s=50
            )

        plt.title("Aggregate Transport Map (All Depots)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        png_path = os.path.join(self.output_dir, "Aggregate_Map.png")
        plt.savefig(png_path)
        plt.close()

        #FOLIUM HTML
        center_lat = df['CusLatitude'].mean()
        center_lon = df['CusLongitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="Cartodb Positron")

        for _, row in df.iterrows():
            depot_id = row['Shipping Point']
            color = depot_color_map.get(depot_id, '#000000')
            
            popup_text = f"""
            <b>Ship To:</b> {row['Ship to']}<br>
            <b>Depot:</b> {depot_id}<br>
            <b>Weight:</b> {row['Total Weight']}<br>
            <b>Volume:</b> {row['Total Volume']}
            """
            
            folium.CircleMarker(
                location=[row['CusLatitude'], row['CusLongitude']],
                radius=5,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)

        legend_html = '''
             <div style="position: fixed; 
             bottom: 50px; left: 50px; width: 150px; height: auto; 
             border:2px solid grey; z-index:9999; font-size:14px;
             background-color:white; opacity: 0.8; padding: 10px;">
             <b>Depot Legend</b><br>
             '''
        for depot, color in depot_color_map.items():
            legend_html += f'<i style="background:{color};width:10px;height:10px;float:left;margin-right:5px;margin-top:4px;"></i>Depot {depot}<br>'
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))

        html_path = os.path.join(self.output_dir, "Aggregate_Map.html")
        m.save(html_path)

    def plot_individual(self, file_path):
        filename = os.path.basename(file_path)
        depot_name = filename.replace("Split_TransportOrder_", "").replace(".csv", "")
        print(f">>> Đang vẽ biểu đồ chi tiết cho Depot: {depot_name}...")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Cannot read file {file_path}: {e}")
            return

        df = df.dropna(subset=['CusLat', 'CusLong'])

        if df.empty:
            print(f"File {filename} has no valid coordinates")
            return

        # MATPLOTLOB PNG
        plt.figure(figsize=(10, 8))
        
        plt.scatter(
            df['CusLong'], 
            df['CusLat'], 
            c='blue', 
            label='Customer', 
            alpha=0.6,
            edgecolors='k'
        )
        
        depot_row = df.iloc[0]
        if pd.notna(depot_row['DepotLat']) and pd.notna(depot_row['DepotLong']):
            plt.scatter(
                depot_row['DepotLong'], 
                depot_row['DepotLat'], 
                c='red', 
                marker='^', 
                s=200, 
                label='Depot Location'
            )

        plt.title(f"Route Map - Depot {depot_name}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True)

        png_path = os.path.join(self.output_dir, f"Individual_{depot_name}.png")
        plt.savefig(png_path)
        plt.close()
        print(f"    -> Đã lưu PNG: {png_path}")

        # FOLIUM HTML
        center_lat = df['CusLat'].mean()
        center_lon = df['CusLong'].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        if pd.notna(depot_row['DepotLat']) and pd.notna(depot_row['DepotLong']):
            folium.Marker(
                [depot_row['DepotLat'], depot_row['DepotLong']],
                popup=f"<b>DEPOT:</b> {depot_name}",
                icon=folium.Icon(color='red', icon='home', prefix='fa')
            ).add_to(m)

        for _, row in df.iterrows():
            popup_text = f"""
            <b>Cust:</b> {row['Customer']}<br>
            <b>HUs:</b> {row['HUs']}<br>
            <b>KGM:</b> {row['KGM']}
            """
            folium.CircleMarker(
                location=[row['CusLat'], row['CusLong']],
                radius=4,
                color='blue',
                fill=True,
                fill_color='blue',
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(m)

        html_path = os.path.join(self.output_dir, f"Individual_{depot_name}.html")
        m.save(html_path)
        print(f"    -> Đã lưu HTML: {html_path}")

if __name__ == "__main__":
    aggregate_file = "AggregateTransportOrder7.csv"
    
    f1 = "Split_TransportOrder_2510.csv"
    f2 = "Split_TransportOrder_2513.csv"
    f3 = "Split_TransportOrder_2522.csv"
    f4 = "Split_TransportOrder_2524.csv"
    f5 = "Split_TransportOrder_2550.csv"

    split_files = [f1,f2,f3,f4,f5]

    viz = TransportVisualizer(output_dir="visualizations")

    if os.path.exists(aggregate_file):
        viz.plot_aggregate(aggregate_file)
    else:
        print(f"Cannot find file: {aggregate_file}")

    if split_files:
        for f in split_files:
            viz.plot_individual(f)
    else:
        print("Cannot find any split file.")