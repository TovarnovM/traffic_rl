"""Simulation parameter schema and validation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationParams:
    num_lanes: int
    road_length: int
    vmax_default: int = 5
    vmax_controlled: int = 6
    G: int = 15
    q: float = 0.99
    r: float = 0.99
    S: int = 2
    P1: float = 0.999
    P2: float = 0.99
    P3: float = 0.98
    P4: float = 0.01
    p_lane_change: float = 0.5

    def __post_init__(self) -> None:
        self._validate_int("num_lanes", self.num_lanes, min_value=1)
        self._validate_int("road_length", self.road_length, min_value=1)
        self._validate_int("vmax_default", self.vmax_default, min_value=0)
        self._validate_int("vmax_controlled", self.vmax_controlled, min_value=0)
        self._validate_int("G", self.G, min_value=0)
        self._validate_int("S", self.S, min_value=1)

        self._validate_prob("q", self.q)
        self._validate_prob("r", self.r)
        self._validate_prob("P1", self.P1)
        self._validate_prob("P2", self.P2)
        self._validate_prob("P3", self.P3)
        self._validate_prob("P4", self.P4)
        self._validate_prob("p_lane_change", self.p_lane_change)

    @staticmethod
    def _validate_int(name: str, value: int, *, min_value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int, got {type(value).__name__}")
        if value < min_value:
            comparator = ">" if min_value == 1 else ">="
            target = 0 if min_value == 1 else min_value
            raise ValueError(f"{name} must be {comparator} {target}, got {value}")

    @staticmethod
    def _validate_prob(name: str, value: float) -> None:
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a float in [0.0, 1.0], got {type(value).__name__}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")


def max_supported_velocity(params: SimulationParams) -> int:
    return max(params.vmax_default, params.vmax_controlled)
