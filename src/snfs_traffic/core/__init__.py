"""Core state and parameter schemas for array-oriented simulation."""

from .params import SimulationParams, max_supported_velocity
from .indexing import (
    INDEX_DTYPE,
    MISSING_GAP,
    MISSING_INDEX,
    build_body_occupancy,
    build_lane_order,
    build_occupancy,
    compute_neighbors,
)
from .state import TrafficState, empty_state, validate_state
from .step_reference import step_longitudinal_reference, step_reference
from .lane_change_reference import step_lane_change_reference
from .invariants import validate_runtime_invariants
from .backend import StepBackend, ReferenceBackend, get_reference_backend
from .roles import PRIORITY_VEH_TYPE, high_speed_vehicle_mask, priority_vehicle_mask

__all__ = [
    "SimulationParams",
    "TrafficState",
    "empty_state",
    "max_supported_velocity",
    "validate_state",
    "INDEX_DTYPE",
    "MISSING_GAP",
    "MISSING_INDEX",
    "build_occupancy",
    "build_body_occupancy",
    "build_lane_order",
    "compute_neighbors",
    "step_longitudinal_reference",
    "step_reference",
    "step_lane_change_reference",
    "validate_runtime_invariants",
    "StepBackend",
    "ReferenceBackend",
    "get_reference_backend",
    "PRIORITY_VEH_TYPE",
    "high_speed_vehicle_mask",
    "priority_vehicle_mask",
]
