"""Core state and parameter schemas for array-oriented simulation."""

from .params import SimulationParams, max_supported_velocity
from .state import TrafficState, empty_state, validate_state

__all__ = [
    "SimulationParams",
    "TrafficState",
    "empty_state",
    "max_supported_velocity",
    "validate_state",
]
