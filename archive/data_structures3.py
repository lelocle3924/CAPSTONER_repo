import numpy as np
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# THUẤN ĐỂ DEFINE CÁC BIẾN ĐÓ GIÁ TRỊ LÀ GÌ THÔI
# TÍNH NHƯ THẾ NÀO THÌ DO EXTRACT_FEATURES BÊN ENV LO

#ver 3: add utilization
# 3.1: thêm max_route_duration

# ==========================================
# PART 1: VEHICLE & FLEET CONFIGURATION
# ==========================================
@dataclass(frozen=True)
class VehicleType:
    """
    Represents a specific type of vehicle in the heterogeneous fleet.
    Immutable configuration loaded from config/input files.
    """
    type_id: int            # Unique identifier (0, 1, 2...)
    name: str               # e.g., "MC", "4w"
    capacity_kg: float      # Max weight (kg)
    capacity_cbm: float     # Max volume (m3)
    speed_kmh: float        # Avg speed (km/h)
    fixed_cost: float       # One-time cost per trip (Currency units)
    cost_per_km: float      # Variable cost per km (Currency units)   -> gas
    cost_per_hour: float    # Variable cost per hour (Currency units) -> driver
    count: int              # Max available vehicles

# ==========================================
# PART 2: PROBLEM DATA (Static Context)
# ==========================================
@dataclass
class ProblemData:
    """
    Container for all static data. Read-only during optimization.
    """
    # --- Matrices ---
    dist_matrix: np.ndarray # (N, N) Meters
    super_time_matrix: np.ndarray # (num_vehicle_types,N, N) Minutes

    # --- Node Info (Index 0 = Depot) ---
    node_ids: List[str]     # ID Mapping
    coords: np.ndarray      # (N, 2) [Lat, Lon]
    demands_kg: np.ndarray  # (N,)
    demands_cbm: np.ndarray # (N,)
    time_windows: np.ndarray # (N, 2) Minutes from start
    service_times: np.ndarray # (N,) Minutes
    allowed_vehicles: List[List[int]] # Node constraints

    vehicle_types: List[VehicleType]

    max_route_duration: float = 1440.0 # minutes = 24 hours

    _id_to_index: Dict[str, int] = field(default_factory=dict, repr=False)

    _static_tw_tightness: float = field(init=False, default=0.0)
    _static_spatial_density: float = field(init=False, default=0.0)

    def __post_init__(self):
        for idx, node_id in enumerate(self.node_ids):
            self._id_to_index[str(node_id)] = idx
            
        # --- CALCULATE STATIC FEATURES IMMEDIATELY ---
        self._calculate_static_features()

    def _calculate_static_features(self):
        # 1. TW Tightness
        # width = End - Start. Avoid division by zero.
        widths = self.time_windows[:, 1] - self.time_windows[:, 0]
        safe_widths = np.maximum(widths, 1.0) # Min width 1 min
        
        # Chỉ tính cho Customer (bỏ Depot index 0)
        cust_service = self.service_times[1:]
        cust_widths = safe_widths[1:]
        
        if len(cust_widths) > 0:
            # Ratio: Service / Width. (e.g. Service 10m, Width 20m -> 0.5)
            # Càng gần 1 càng chặt.
            self._static_tw_tightness = float(np.mean(cust_service / cust_widths))
        
        # 2. Spatial Density (Based on Dist Matrix)
        cust_dist = self.dist_matrix[1:, 1:]
        
        if cust_dist.shape[0] > 1:
            # Thay số 0 ở đường chéo bằng vô cực để tìm min không phải chính nó
            temp_dist = cust_dist.copy()
            np.fill_diagonal(temp_dist, np.inf)
            
            # Tìm khoảng cách đến hàng xóm gần nhất cho mỗi node
            min_dists = np.min(temp_dist, axis=1)
            avg_min_dist = np.mean(min_dists)
            
            # Density = 1000 / AvgDist (Để số không quá nhỏ, đơn vị node/km)
            # Nếu AvgDist = 500m -> Density = 2.0
            self._static_spatial_density = 1000.0 / (avg_min_dist + 1.0)
            
    @property
    def static_tw_tightness(self) -> float:
        return self._static_tw_tightness

    @property
    def static_spatial_density(self) -> float:
        return self._static_spatial_density

    @property
    def num_nodes(self) -> int:
        """Total number of nodes (including depot)."""
        return len(self.node_ids)

    def get_travel_time(self, from_node: int, to_node: int, vehicle_type_id: int) -> float:
        return self.super_time_matrix[vehicle_type_id, from_node, to_node]


# ==========================================
# PART 3: SOLUTION REPRESENTATION (Route & State)
# ==========================================
@dataclass
class Route:
    """
    Represents a single route performed by a specific vehicle.
    """
    vehicle_type: VehicleType 
    node_sequence: List[int]  
    
    # --- Cached Metrics ---
    total_dist_meters: float = 0.0
    total_duration_min: float = 0.0
    total_wait_time_min: float = 0.0
    total_load_kg: float = 0.0
    total_load_cbm: float = 0.0
    centroid_lat: float = 0.0
    centroid_lon: float = 0.0
    
    # --- Feasibility ---
    is_time_feasible: bool = True
    is_capacity_feasible: bool = True
    is_duration_feasible: bool = True
    is_preference_feasible: bool = True
    
    def update_centroid(self, data):
        """Cập nhật centroid dựa trên các node trong route"""
        if not self.node_sequence:
            # Nếu rỗng, lấy toạ độ depot
            self.centroid_lat = data.coords[0][0]
            self.centroid_lon = data.coords[0][1]
        else:
            # Lấy trung bình cộng toạ độ các node
            # data.coords shape (N, 2)
            nodes = self.node_sequence
            coords = data.coords[nodes]
            mean_coords = np.mean(coords, axis=0)
            self.centroid_lat = mean_coords[0]
            self.centroid_lon = mean_coords[1]

    def clone(self) -> 'Route':
        """
        Fast clone method to avoid deepcopy overhead.
        We only copy mutable structures (node_sequence).
        VehicleType is immutable (dataclass frozen) so we reference it.
        """
        return Route(
            vehicle_type=self.vehicle_type,
            node_sequence=self.node_sequence[:], # Fast slicing copy
            total_dist_meters=self.total_dist_meters,
            total_duration_min=self.total_duration_min,
            total_wait_time_min=self.total_wait_time_min,
            total_load_kg=self.total_load_kg,
            total_load_cbm=self.total_load_cbm,
            centroid_lat=self.centroid_lat,
            centroid_lon=self.centroid_lon,
            is_time_feasible=self.is_time_feasible,
            is_capacity_feasible=self.is_capacity_feasible,
            is_duration_feasible=self.is_duration_feasible,
            is_preference_feasible=self.is_preference_feasible
        )

    @property
    def cost(self) -> float:
        v = self.vehicle_type
        return v.fixed_cost + (self.total_dist_meters/1000.0 * v.cost_per_km) + (self.total_duration_min/60.0 * v.cost_per_hour)
    @property
    def capacity_utilization(self) -> float:
        if self.vehicle_type.capacity_kg <= 0 or self.vehicle_type.capacity_cbm <= 0: return 0.0
        util_kg = self.total_load_kg / self.vehicle_type.capacity_kg
        util_cbm = self.total_load_cbm / self.vehicle_type.capacity_cbm
        return max(util_kg, util_cbm)

    @property
    def is_feasible(self) -> bool:
        return self.is_time_feasible and self.is_capacity_feasible and self.is_duration_feasible and self.is_preference_feasible

    @property
    def wait_time_ratio(self) -> float:
        if self.total_duration_min < 1e-6: return 0.0
        return self.total_wait_time_min / self.total_duration_min

@dataclass
class RvrpState:
    """
    Rich VRP State. Represents a complete solution.
    """
    routes: List[Route]       
    unassigned: List[int]     

    def copy(self) -> 'RvrpState':
        """
        Optimized copy method.
        Avoids copy.deepcopy which is extremely slow.
        """
        # Sử dụng hàm clone() đã tối ưu của Route
        new_routes = [r.clone() for r in self.routes]
        # Unassigned là list int, slice copy là đủ nhanh
        return RvrpState(new_routes, self.unassigned[:])

    def objective(self) -> float:
        """
        Total Cost = Sum(Route Costs) + Penalty(Unassigned).
        Route Cost implicitly includes Fixed + Variable costs via Route.cost property.
        """
        operational_cost = sum(r.cost for r in self.routes)
        util_penalty = 0
        for r in self.routes:
            if r.capacity_utilization < 0.5:
                util_penalty += (0.5 - r.capacity_utilization) * r.vehicle_type.fixed_cost * 5.0
        return operational_cost + util_penalty

    @property
    def min_capacity_utilization(self) -> float:
        if not self.routes: return 0.0
        return min(r.capacity_utilization for r in self.routes)

    @property
    def max_capacity_utilization(self) -> float:
        if not self.routes: return 0.0
        return max(r.capacity_utilization for r in self.routes)

    @property
    def mean_capacity_utilization(self) -> float:
        if not self.routes: return 0.0
        return float(np.mean([r.capacity_utilization for r in self.routes]))
    
    @property
    def max_wait_time_ratio(self) -> float:
        if not self.routes: return 0.0
        return max(r.wait_time_ratio for r in self.routes)

# ==========================================
# PART 4: PPO AGENT STATE (Observation)
# ==========================================
@dataclass
class PPOState:
    """
    Fixed-size feature vector for PPO input.
    """
    # --- Search Status ---
    search_progress: float      
    stagnation_norm: float      
    best_cost_norm: float       # Target for reward
    current_cost_norm: float    
    improvement_history: float  
    
    # --- Instance Stats ---
    demands_mean: float         
    demands_std: float          
    tw_width_mean: float        
    tw_tightness: float         
    spatial_density: float      
    
    # --- Solution Stats (Dynamic) ---
    min_cap_utilization: float  
    mean_cap_utilization: float # Target for reward
    max_cap_utilization: float
    max_wait_time_ratio: float

    num_routes_norm: float      
    num_unassigned_norm: float      
    
    # --- Operator History ---
    destroy_probs: np.ndarray   
    repair_probs: np.ndarray    
    
    def to_array(self) -> np.ndarray:
        scalars = np.array([
            self.search_progress,
            self.stagnation_norm,
            self.best_cost_norm,
            self.current_cost_norm,
            self.improvement_history,
            self.demands_mean,
            self.demands_std,
            self.tw_width_mean,
            self.tw_tightness,
            self.spatial_density,
            self.min_cap_utilization,
            self.mean_cap_utilization,
            self.max_cap_utilization,
            self.max_wait_time_ratio,
            self.num_routes_norm,
            self.num_unassigned_norm
        ], dtype=np.float32)
        
        vec_destroy_probs = self.destroy_probs.astype(np.float32)
        vec_repair_probs = self.repair_probs.astype(np.float32)
        
        return np.concatenate([scalars, vec_destroy_probs, vec_repair_probs])

    @staticmethod
    def get_observation_size(n_destroy: int, n_repair: int) -> int:
        NUM_SCALARS = 16 # Updated count (13 + 2 new util metrics)
        return NUM_SCALARS + n_destroy + n_repair