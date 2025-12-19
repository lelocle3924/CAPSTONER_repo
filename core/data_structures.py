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
    time_matrix: np.ndarray # (N, N) Minutes

    # --- Node Info (Index 0 = Depot) ---
    node_ids: List[str]     # ID Mapping
    coords: np.ndarray      # (N, 2) [Lat, Lon]
    demands_kg: np.ndarray  # (N,)
    demands_cbm: np.ndarray # (N,)
    time_windows: np.ndarray # (N, 2) Minutes from start
    service_times: np.ndarray # (N,) Minutes
    allowed_vehicles: List[List[int]] # Node constraints

    # --- Fleet Info ---
    vehicle_types: List[VehicleType]

    # --- Global constraint for route max time ---
    max_route_duration: float = 600.0 # minutes = 10 hours

    # --- Helpers ---
    _id_to_index: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for idx, node_id in enumerate(self.node_ids):
            self._id_to_index[str(node_id)] = idx

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

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
    
    # --- Feasibility ---
    is_time_feasible: bool = True
    is_capacity_feasible: bool = True
    is_duration_feasible: bool = True
    
    @property
    def cost(self) -> float:
        """
        Economic Cost = Fixed Cost + (Distance_km * Cost_per_km)
        """
        dist_km = self.total_dist_meters / 1000.0
        time_hour = self.total_duration_min / 60.0
        return self.vehicle_type.fixed_cost + (dist_km * self.vehicle_type.cost_per_km) + (time_hour * self.vehicle_type.cost_per_hour)

    @property
    def capacity_utilization(self) -> float:
        """
        Returns the higher utilization rate between Weight and Volume.
        Value between 0.0 and 1.0 (or >1.0 if infeasible).
        """
        # Avoid division by zero
        if self.vehicle_type.capacity_kg <= 0 or self.vehicle_type.capacity_cbm <= 0:
            print("[DATA STRUCTURE ERROR]:Capacity is zero or negative")
            return 0.0
            
        util_kg = self.total_load_kg / self.vehicle_type.capacity_kg
        util_cbm = self.total_load_cbm / self.vehicle_type.capacity_cbm
        
        return max(util_kg, util_cbm)

    @property
    def is_feasible(self) -> bool:
        return self.is_time_feasible and self.is_capacity_feasible and self.is_duration_feasible

    @property
    def wait_time_ratio(self) -> float:
        return self.total_wait_time_min / self.total_duration_min

@dataclass
class RvrpState:
    """
    Rich VRP State. Represents a complete solution.
    """
    routes: List[Route]       
    unassigned: List[int]     

    def copy(self) -> 'RvrpState':
        """Deep copy of the state."""
        new_routes = [
            Route(
                vehicle_type=r.vehicle_type,
                node_sequence=r.node_sequence[:],
                total_dist_meters=r.total_dist_meters,
                total_duration_min=r.total_duration_min,
                total_wait_time_min=r.total_wait_time_min,
                total_load_kg=r.total_load_kg,
                total_load_cbm=r.total_load_cbm,
                is_time_feasible=r.is_time_feasible,
                is_capacity_feasible=r.is_capacity_feasible,
                is_duration_feasible=r.is_duration_feasible
            ) for r in self.routes
        ]
        return RvrpState(new_routes, self.unassigned[:])

    def objective(self) -> float:
        """
        Total Cost = Sum(Route Costs) + Penalty(Unassigned).
        Route Cost implicitly includes Fixed + Variable costs via Route.cost property.
        """
        operational_cost = sum(r.cost for r in self.routes)
        
        unassigned_penalty = len(self.unassigned) * 1e9 # phạt cho khách không đc serve

        underutilization_penalty = 0 * self.mean_capacity_utilization  # phạt cho capacity bị dư -> hiện tại cho bằng 0 (trừ khi muốn đổi utilization vào đây)
        
        return operational_cost + unassigned_penalty + underutilization_penalty

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
        # Dùng numpy mean cho chính xác và nhanh
        return float(np.mean([r.capacity_utilization for r in self.routes]))

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