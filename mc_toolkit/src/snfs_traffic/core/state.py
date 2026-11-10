"""Traffic state array schema and validation."""

from dataclasses import dataclass

import numpy as np

from .params import SimulationParams
from .roles import high_speed_vehicle_mask
from .types import (
    BEHAVIOR_ID_DTYPE,
    BOOL_DTYPE,
    LANE_DELTA_DTYPE,
    LANE_DTYPE,
    LENGTH_DTYPE,
    POSITION_DTYPE,
    VEHICLE_ID_DTYPE,
    VEHICLE_TYPE_DTYPE,
    VELOCITY_DTYPE,
)


@dataclass(slots=True)
class TrafficState:
    vehicle_id: np.ndarray
    lane: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    length: np.ndarray
    veh_type: np.ndarray
    behavior_id: np.ndarray
    alive: np.ndarray
    last_lane_delta: np.ndarray
    changed_lane: np.ndarray
    controlled: np.ndarray

    @property
    def n_vehicles(self) -> int:
        return int(self.vehicle_id.shape[0])

    def copy(self) -> "TrafficState":
        return TrafficState(
            vehicle_id=self.vehicle_id.copy(),
            lane=self.lane.copy(),
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            length=self.length.copy(),
            veh_type=self.veh_type.copy(),
            behavior_id=self.behavior_id.copy(),
            alive=self.alive.copy(),
            last_lane_delta=self.last_lane_delta.copy(),
            changed_lane=self.changed_lane.copy(),
            controlled=self.controlled.copy(),
        )


def validate_state(state: TrafficState, params: SimulationParams) -> None:
    spec = {
        "vehicle_id": VEHICLE_ID_DTYPE,
        "lane": LANE_DTYPE,
        "pos": POSITION_DTYPE,
        "vel": VELOCITY_DTYPE,
        "length": LENGTH_DTYPE,
        "veh_type": VEHICLE_TYPE_DTYPE,
        "behavior_id": BEHAVIOR_ID_DTYPE,
        "alive": BOOL_DTYPE,
        "last_lane_delta": LANE_DELTA_DTYPE,
        "changed_lane": BOOL_DTYPE,
        "controlled": BOOL_DTYPE,
    }

    expected_shape = None
    for field, expected_dtype in spec.items():
        arr = getattr(state, field)
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"{field} must be a numpy.ndarray")
        if arr.ndim != 1:
            raise ValueError(f"{field} must be one-dimensional, got shape {arr.shape}")
        if arr.dtype != np.dtype(expected_dtype):
            raise ValueError(f"{field} must have dtype {np.dtype(expected_dtype)}, got {arr.dtype}")
        if not arr.flags.c_contiguous:
            raise ValueError(f"{field} must be C-contiguous")
        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            raise ValueError(f"{field} shape {arr.shape} does not match expected shape {expected_shape}")

    alive_mask = state.alive

    alive_vehicle_id = state.vehicle_id[alive_mask]
    if np.any(alive_vehicle_id < 0):
        raise ValueError("vehicle_id has negative values for alive vehicles")
    if np.unique(alive_vehicle_id).size != alive_vehicle_id.size:
        raise ValueError("vehicle_id must be unique among alive vehicles")

    alive_lane = state.lane[alive_mask]
    if np.any((alive_lane < 0) | (alive_lane >= params.num_lanes)):
        raise ValueError("lane out of range for alive vehicles")

    alive_pos = state.pos[alive_mask]
    if np.any((alive_pos < 0) | (alive_pos >= params.road_length)):
        raise ValueError("pos out of range for alive vehicles")

    alive_vel = state.vel[alive_mask]
    if np.any(alive_vel < 0):
        raise ValueError("vel out of range for alive vehicles")

    alive_high_speed = high_speed_vehicle_mask(state)[alive_mask]
    default_speed_alive_vel = alive_vel[~alive_high_speed]
    if np.any(default_speed_alive_vel > params.vmax_default):
        raise ValueError("vel out of range for uncontrolled non-priority alive vehicles")

    high_speed_alive_vel = alive_vel[alive_high_speed]
    if np.any(high_speed_alive_vel > params.vmax_controlled):
        raise ValueError("vel out of range for controlled or priority alive vehicles")

    alive_length = state.length[alive_mask]
    if np.any(alive_length < 1):
        raise ValueError("length must be >= 1 for alive vehicles")

    alive_lane_delta = state.last_lane_delta[alive_mask]
    if np.any(~np.isin(alive_lane_delta, np.array([-1, 0, 1], dtype=LANE_DELTA_DTYPE))):
        raise ValueError("last_lane_delta must be one of {-1, 0, +1} for alive vehicles")

    expected_changed = alive_lane_delta != 0
    if np.any(state.changed_lane[alive_mask] != expected_changed):
        raise ValueError("changed_lane must equal (last_lane_delta != 0) for alive vehicles")


def empty_state(n_vehicles: int) -> TrafficState:
    if isinstance(n_vehicles, bool) or not isinstance(n_vehicles, int) or n_vehicles < 0:
        raise ValueError("n_vehicles must be an int >= 0")

    zeros_i16 = np.zeros(n_vehicles, dtype=np.int16)
    return TrafficState(
        vehicle_id=np.arange(n_vehicles, dtype=VEHICLE_ID_DTYPE),
        lane=np.zeros(n_vehicles, dtype=LANE_DTYPE),
        pos=np.zeros(n_vehicles, dtype=POSITION_DTYPE),
        vel=np.zeros(n_vehicles, dtype=VELOCITY_DTYPE),
        length=np.ones(n_vehicles, dtype=LENGTH_DTYPE),
        veh_type=zeros_i16.copy(),
        behavior_id=zeros_i16.copy(),
        alive=np.ones(n_vehicles, dtype=BOOL_DTYPE),
        last_lane_delta=np.zeros(n_vehicles, dtype=LANE_DELTA_DTYPE),
        changed_lane=np.zeros(n_vehicles, dtype=BOOL_DTYPE),
        controlled=np.zeros(n_vehicles, dtype=BOOL_DTYPE),
    )
