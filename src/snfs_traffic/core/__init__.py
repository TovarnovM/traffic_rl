"""Core state and parameter schemas for array-oriented simulation."""

from .params import SimulationParams, max_supported_velocity
from .indexing import (
    INDEX_DTYPE,
    MISSING_GAP,
    MISSING_INDEX,
    build_lane_order,
    build_occupancy,
    compute_neighbors,
)
from .state import TrafficState, empty_state, validate_state
from .step_reference import step_longitudinal_reference, step_reference
from .lane_change_reference import step_lane_change_reference

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
    "build_lane_order",
    "compute_neighbors",
    "step_longitudinal_reference",
    "step_reference",
    "step_lane_change_reference",
]
