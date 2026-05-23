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
]
