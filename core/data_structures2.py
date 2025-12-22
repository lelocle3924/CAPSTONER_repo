import numpy as np
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Ver 2: aggregate most of dataclasses into one file (from State.py and solution.py)

# ==========================================
# PART 1: VEHICLE & FLEET CONFIGURATION
# ==========================================
@dataclass(frozen=True)
class VehicleType:
    """
    Represents a specific type of vehicle in the heterogeneous fleet.
    Immutable configuration loaded from config/input files.
    """
    type_id: int            # Unique identifier for the vehicle type (0, 1, 2...)
    name: str               # Human-readable name (e.g., "MC", "4w", "10w")
    capacity_kg: float      # Maximum weight capacity in kilograms
    capacity_cbm: float     # Maximum volume capacity in cubic meters
    speed_kmh: float        # Average speed in km/h (used for travel time calculation)
    fixed_cost: float       # One-time cost for using this vehicle (Driver fee, depreciation)
    cost_per_km: float      # Variable cost per kilometer traveled (Fuel, maintenance)
    count: int              # Maximum available number of vehicles of this type

# ==========================================
# PART 2: PROBLEM DATA (Static Context)
# ==========================================
@dataclass
class ProblemData:
    """
    Container for all static data of the VRP instance. 
    Passed to operators and states to check constraints and calculate costs.
    """
    # --- Matrices ---
    dist_matrix: np.ndarray # Shape (N, N). Unit: Meters. Distance between all nodes.
    time_matrix: np.ndarray # Shape (N, N). Unit: Minutes. Travel time between all nodes.

    node_ids: List[str]     # Mapping from internal Index to Real Customer ID (e.g., 0 -> "2524")
    coords: np.ndarray      # Shape (N, 2). Columns: [Latitude, Longitude].
    
    demands_kg: np.ndarray  # Shape (N,). Weight demand for each node.
    demands_cbm: np.ndarray # Shape (N,). Volume demand for each node.
    
    time_windows: np.ndarray # Shape (N, 2). Columns: [Start_Min, End_Min]. Normalized to minutes from day start.
    service_times: np.ndarray # Shape (N,). Unit: Minutes. Dwell time at each node.
    
    allowed_vehicles: List[List[int]] # List of size N. Each element is a list of allowed vehicle_type_ids for that node.

    vehicle_types: List[VehicleType] # List of all available vehicle configurations.

    _id_to_index: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Automatically create the ID mapping after initialization."""
        for idx, node_id in enumerate(self.node_ids):
            self._id_to_index[str(node_id)] = idx

    @property
    def num_nodes(self) -> int:
        """Total number of nodes (including depot)."""
        return len(self.node_ids)

# ==========================================
# PART 3: SOLUTION REPRESENTATION (Route & State)
# ==========================================
@dataclass
class Route:
    """
    Represents a single route performed by a specific vehicle.
    Contains sequence of visits and cached statistics.
    """
    vehicle_type: VehicleType # The vehicle configuration assigned to this route
    node_sequence: List[int]  # Sequence of customer indices (excluding start/end depot implied)
    
    # --- Cached Metrics (Calculated upon creation/modification) ---
    total_dist_meters: float = 0.0
    total_time_min: float = 0.0
    total_load_kg: float = 0.0
    total_load_cbm: float = 0.0
    
    # --- Feasibility Flags ---
    is_time_feasible: bool = True
    is_capacity_feasible: bool = True
    
    @property
    def cost(self) -> float:
        """
        Calculate total cost of this route: Fixed Cost + (Distance * Cost/km).
        Distance is converted to km (meters / 1000).
        """
        dist_km = self.total_dist_meters / 1000.0
        return self.vehicle_type.fixed_cost + (dist_km * self.vehicle_type.cost_per_km)

@dataclass
class CvrptwState:
    """
    Represents a complete solution to the Heterogeneous Fleet VRPTW.
    Replaces the old list-based state.
    """
    routes: List[Route]       # List of active routes
    unassigned: List[int]     # List of unassigned customer indices
    
    # Optional: Reference to ProblemData if needed for deep feasibility checks, 
    # but strictly speaking State should be independent or light.
    # Here we assume routes are already populated with correct metrics.

    def copy(self) -> 'CvrptwState':
        """Creates a deep copy of the state."""
        # Deep copy routes because lists are mutable
        new_routes = [
            Route(
                vehicle_type=r.vehicle_type, # VehicleType is frozen (safe to reference)
                node_sequence=r.node_sequence[:], # Copy list
                total_dist_meters=r.total_dist_meters,
                total_time_min=r.total_time_min,
                total_load_kg=r.total_load_kg,
                total_load_cbm=r.total_load_cbm,
                is_time_feasible=r.is_time_feasible,
                is_capacity_feasible=r.is_capacity_feasible
            ) for r in self.routes
        ]
        return CvrptwState(new_routes, self.unassigned[:])

    def objective(self) -> float:
        """
        The primary objective function minimized by ALNS.
        Sum of all route costs + Penalties for unassigned customers.
        """
        route_costs = sum(r.cost for r in self.routes)
        
        # Penalty for unassigned customers (Should be very high)
        # Using a simplistic penalty here, can be tuned.
        unassigned_penalty = len(self.unassigned) * 1e6 
        
        return route_costs + unassigned_penalty

# ==========================================
# PART 4: PPO AGENT STATE (Observation)
# ==========================================
@dataclass
class PPOState:
    """
    Fixed-size feature vector representing the current state of the search.
    This acts as the observation for the PPO Neural Network.
    """
    # --- SCALARS: Search Status ---
    search_progress: float      # [0, 1] Current iteration / Max iterations
    stagnation_norm: float      # [0, 1] Current stagnation depth (steps without improvement)
    best_cost_norm: float       # Best objective value found so far (Normalized)
    current_cost_norm: float    # Current objective value (Normalized)
    improvement_history: float  # Moving average of improvement rate over recent steps
    
    # --- SCALARS: Instance Characteristics (Statistical) ---
    demands_mean: float         # Mean of normalized demands (kg)
    demands_std: float          # Standard deviation of demands (volatility)
    tw_width_mean: float        # Mean time window width (tightness of schedule)
    tw_tightness: float         # Fraction of customers with very tight time windows
    spatial_density: float      # Proxy for node density (e.g., avg distance to depot)
    
    # --- SCALARS: Solution Status (Dynamic) ---
    avg_capacity_utilization: float # Average vehicle fill rate (0 = empty, 1 = full)
    num_routes_norm: float          # Number of active vehicles / Total fleet size (estimate)
    num_unassigned_norm: float      # Number of unassigned customers / Total customers
    
    # --- VECTORS: Operator History ---
    destroy_probs: np.ndarray   # Shape (n_destroy,). Normalized usage frequency of destroy ops.
    repair_probs: np.ndarray    # Shape (n_repair,). Normalized usage frequency of repair ops.
    
    def to_array(self) -> np.ndarray:
        """
        Flatten all features into a single 1D float32 array.
        This array is fed directly into the PPO Policy Network.
        """
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
            self.avg_capacity_utilization,
            self.num_routes_norm,
            self.num_unassigned_norm
        ], dtype=np.float32)
        
        # Ensure vectors are float32
        vec_destroy = self.destroy_probs.astype(np.float32)
        vec_repair = self.repair_probs.astype(np.float32)
        
        # Concatenate everything
        return np.concatenate([scalars, vec_destroy, vec_repair])

    @staticmethod
    def get_observation_size(n_destroy: int, n_repair: int) -> int:
        """
        Calculate the total size of the observation vector.
        Currently: 13 scalars + n_destroy + n_repair.
        """
        NUM_SCALARS = 13
        return NUM_SCALARS + n_destroy + n_repair