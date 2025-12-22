# file: core/visualizer.py ver 1

import folium
import os
import random
from typing import List, Dict
from core.data_structures import RvrpState, ProblemData, Route

class RouteVisualizer:
    def __init__(self, output_dir="results/maps"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define color palette based on vehicle names (heuristic)
        self.color_map = {
            'MC': 'green',       # Motorbike
            'AUV': 'orange',     # Small van
            '4w': 'blue',        # 4 Wheel
            '6w': 'purple',      # 6 Wheel
            '10w': 'red',        # Big truck
            '40ft': 'black'      # Container
        }
        self.fallback_colors = ['cadetblue', 'darkgreen', 'darkblue', 'darkred', 'gray']

    def _get_color(self, vehicle_name: str) -> str:
        # Simple fuzzy match or direct lookup
        for key, color in self.color_map.items():
            if key in vehicle_name:
                return color
        return random.choice(self.fallback_colors)

    def visualize_solution(self, solution: RvrpState, data: ProblemData, filename: str = "optimized_route.html"):
        """
        Generates an HTML map of the solution.
        """
        # 1. Base Map centered at Depot
        depot_coords = data.coords[0]
        m = folium.Map(location=[depot_coords[0], depot_coords[1]], zoom_start=12, tiles='CartoDB positron')

        # 2. Draw Nodes (Depot & Customers)
        # Depot
        folium.Marker(
            location=[depot_coords[0], depot_coords[1]],
            popup="<b>DEPOT</b>",
            icon=folium.Icon(color='black', icon='home', prefix='fa')
        ).add_to(m)

        # 3. Draw Routes
        layers = {}
        
        for i, route in enumerate(solution.routes):
            v_name = route.vehicle_type.name
            color = self._get_color(v_name)
            
            # Prepare Path Coordinates
            # Note: route.node_sequence does not include depot, we need to add it for visualization
            path_indices = [0] + route.node_sequence + [0]
            path_coords = [data.coords[idx] for idx in path_indices]
            
            # Create Feature Group for Layer Control
            layer_name = f"R{i+1}: {v_name} ({len(route.node_sequence)} stops)"
            fg = folium.FeatureGroup(name=layer_name)
            
            # Draw PolyLine
            folium.PolyLine(
                locations=path_coords,
                color=color,
                weight=3,
                opacity=0.8,
                tooltip=f"Route {i+1} ({v_name})"
            ).add_to(fg)
            
            # Add markers for customers in this route
            for seq_idx, node_idx in enumerate(route.node_sequence):
                # Calculate simple stats for popup
                demand_str = f"{data.demands_kg[node_idx]:.0f}kg"
                tw_str = f"{int(data.time_windows[node_idx][0])}-{int(data.time_windows[node_idx][1])}"
                
                folium.CircleMarker(
                    location=[data.coords[node_idx][0], data.coords[node_idx][1]],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_color='white',
                    fill_opacity=1,
                    popup=folium.Popup(f"<b>Cust {data.node_ids[node_idx]}</b><br>Seq: {seq_idx+1}<br>Dem: {demand_str}<br>TW: {tw_str}", max_width=200)
                ).add_to(fg)
            
            # Route Summary Popup (Click on line)
            summary_html = (
                f"<b>Route {i+1} Summary</b><br>"
                f"Vehicle: {v_name}<br>"
                f"Load: {route.total_load_kg:.1f} / {route.vehicle_type.capacity_kg} kg<br>"
                f"Util: {route.capacity_utilization*100:.1f}%<br>"
                f"Dist: {route.total_dist_meters/1000:.1f} km<br>"
                f"Duration: {route.total_duration_min:.1f} min<br>"
                f"Stops: {len(route.node_sequence)}<br>"
                f"Cost: {route.cost:,.0f}"
            )
            
            # Hack to add popup to PolyLine (folium sometimes limits this, we add a transparent marker in middle)
            mid_point = path_coords[len(path_coords)//2]
            folium.Marker(
                mid_point, 
                icon=folium.DivIcon(html=f"""<div style="font-size:0pt">.</div>"""),
                popup=folium.Popup(summary_html, max_width=250)
            ).add_to(fg)

            fg.add_to(m)

        # 4. Draw Unassigned (if any)
        if solution.unassigned:
            fg_un = folium.FeatureGroup(name="Unassigned Customers", show=True)
            for u_idx in solution.unassigned:
                folium.CircleMarker(
                    location=[data.coords[u_idx][0], data.coords[u_idx][1]],
                    radius=5,
                    color='gray',
                    fill=True,
                    fill_color='red',
                    popup=f"Unassigned: {data.node_ids[u_idx]}"
                ).add_to(fg_un)
            fg_un.add_to(m)

        folium.LayerControl().add_to(m)
        
        save_path = os.path.join(self.output_dir, filename)
        m.save(save_path)
        print(f"  > Map saved to: {save_path}")