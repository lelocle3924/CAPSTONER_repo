#ver 2 - không có cache geometry
# file: core/visualizer.py

import folium
import os
import random
from typing import List, Dict
from core.data_structures import RvrpState, ProblemData, Route
from config import PathConfig

class RouteVisualizer:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir if output_dir else PathConfig.RESULTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        # [UPDATED] Color Mapping Standardized
        self.color_map = {
            'MC': 'green',          # Motorbike
            'AUV': 'orange',        # Van/Small
            '4w': 'blue',           # Small Truck
            '6w': 'purple',         # Medium Truck
            '10w': 'purple',        # Large Truck
            '40ft': 'black'         # Container (Warning color)
        }
        self.fallback_colors = ['cadetblue', 'darkgreen', 'darkblue', 'gray']

    def _get_color(self, vehicle_name: str) -> str:
        # Check strict keys first
        if vehicle_name in self.color_map:
            return self.color_map[vehicle_name]
        # Check partial match
        for key, color in self.color_map.items():
            if key in vehicle_name:
                return color
        return random.choice(self.fallback_colors)

    def visualize_solution(self, solution: RvrpState, data: ProblemData, filename: str = "optimized_route.html"):
        """
        Generates an HTML map of the solution with upgraded UI.
        """
        if not data.coords.any():
            print("  [Visualizer] No coordinates data found.")
            return

        depot_coords = data.coords[0]
        m = folium.Map(location=[depot_coords[0], depot_coords[1]], zoom_start=12, tiles='CartoDB positron')

        # [UPDATED] Depot Icon: Red Warehouse
        folium.Marker(
            location=[depot_coords[0], depot_coords[1]],
            popup="<b>CENTRAL DEPOT</b>",
            icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
        ).add_to(m)

        # Draw Routes
        for i, route in enumerate(solution.routes):
            v_name = route.vehicle_type.name
            color = self._get_color(v_name)
            
            # Path
            path_indices = [0] + route.node_sequence + [0]
            path_coords = [data.coords[idx] for idx in path_indices]
            
            layer_name = f"R{i+1}: {v_name} ({len(route.node_sequence)} stops)"
            fg = folium.FeatureGroup(name=layer_name)
            
            # Draw Line
            folium.PolyLine(
                locations=path_coords,
                color=color,
                weight=3,
                opacity=0.8,
                tooltip=f"Route {i+1} ({v_name})"
            ).add_to(fg)
            
            # [UPDATED] Utilization Warning Logic
            util_percent = route.capacity_utilization * 100
            warning_html = ""
            if util_percent < 50.0:
                warning_html = "<br><b style='color:red;'>⚠️ LOW UTILIZATION</b>"
            
            # Summary Popup
            summary_html = (
                f"<b>Route {i+1} Summary</b><br>"
                f"Vehicle: <b>{v_name}</b><br>"
                f"Load: {route.total_load_kg:.0f} / {route.vehicle_type.capacity_kg:.0f} kg<br>"
                f"Util: <b>{util_percent:.1f}%</b> {warning_html}<br>"
                f"Dist: {route.total_dist_meters/1000:.1f} km<br>"
                f"Duration: {route.total_duration_min:.0f} min<br>"
                f"Stops: {len(route.node_sequence)}<br>"
                f"Cost: {route.cost:,.0f}"
            )
            
            # Invisible marker for popup on line
            mid_point = path_coords[len(path_coords)//2]
            folium.Marker(
                mid_point, 
                icon=folium.DivIcon(html=f"""<div style="font-size:0pt">.</div>"""),
                popup=folium.Popup(summary_html, max_width=250)
            ).add_to(fg)
            
            # Draw Customers
            for seq_idx, node_idx in enumerate(route.node_sequence):
                demand_str = f"{data.demands_kg[node_idx]:.0f}kg"
                tw_str = f"{int(data.time_windows[node_idx][0])}-{int(data.time_windows[node_idx][1])}"
                
                folium.CircleMarker(
                    location=[data.coords[node_idx][0], data.coords[node_idx][1]],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_color='white',
                    fill_opacity=1.0,
                    popup=folium.Popup(f"<b>{data.node_ids[node_idx]}</b><br>Seq: {seq_idx+1}<br>Dem: {demand_str}<br>TW: {tw_str}", max_width=200)
                ).add_to(fg)

            fg.add_to(m)

        # Draw Unassigned
        if solution.unassigned:
            fg_un = folium.FeatureGroup(name="Unassigned", show=True)
            for u_idx in solution.unassigned:
                folium.CircleMarker(
                    location=[data.coords[u_idx][0], data.coords[u_idx][1]],
                    radius=6,
                    color='red',
                    fill=True,
                    fill_color='darkred',
                    popup=f"Unassigned: {data.node_ids[u_idx]}"
                ).add_to(fg_un)
            fg_un.add_to(m)

        folium.LayerControl().add_to(m)
        
        save_path = os.path.join(self.output_dir, filename)
        m.save(save_path)
        print(f"  > [Visualizer] Map saved to: {save_path}")