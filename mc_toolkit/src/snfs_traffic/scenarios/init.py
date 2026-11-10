from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snfs_traffic.core import TrafficState
from snfs_traffic.core.types import (
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

HDV_VEH_TYPE = 0
AV_VEH_TYPE = 1
BUS_VEH_TYPE = 2

HDV_BEHAVIOR_ID = 0
AV_BEHAVIOR_ID = 1
CONTROLLED_AV_BEHAVIOR_ID = 2
BUS_BEHAVIOR_ID = 3
PRIORITY_BEHAVIOR_ID = 4


_MAX_LANE_VALUE = np.iinfo(np.dtype(LANE_DTYPE)).max
_MAX_LENGTH_VALUE = np.iinfo(np.dtype(LENGTH_DTYPE)).max


@dataclass(frozen=True, slots=True)
class VehicleMix:
    av_fraction: float = 0.0
    controlled_fraction: float = 0.0
    bus_fraction: float = 0.0
    bus_length: int = 3

    def __post_init__(self) -> None:
        av_fraction = _validate_probability("av_fraction", self.av_fraction)
        controlled_fraction = _validate_probability("controlled_fraction", self.controlled_fraction)
        bus_fraction = _validate_probability("bus_fraction", self.bus_fraction)
        _validate_positive_int("bus_length", self.bus_length)
        if self.bus_length > _MAX_LENGTH_VALUE:
            raise ValueError(f"bus_length must fit in {np.dtype(LENGTH_DTYPE)}, got {self.bus_length}")

        if av_fraction + controlled_fraction + bus_fraction > 1.0:
            raise ValueError(
                "av_fraction + controlled_fraction + bus_fraction must be <= 1.0"
            )


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _validate_probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a numeric probability in [0.0, 1.0]")
    value_f = float(value)
    if not 0.0 <= value_f <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0]")
    return value_f


def _validate_seed(seed: int | np.integer) -> int | np.integer:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an int or numpy integer")
    return seed


def _assign_vehicle_types(n_vehicles: int, mix: VehicleMix, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    veh_type = np.full(n_vehicles, HDV_VEH_TYPE, dtype=VEHICLE_TYPE_DTYPE)
    behavior_id = np.full(n_vehicles, HDV_BEHAVIOR_ID, dtype=BEHAVIOR_ID_DTYPE)
    controlled = np.zeros(n_vehicles, dtype=BOOL_DTYPE)
    length = np.ones(n_vehicles, dtype=LENGTH_DTYPE)

    controlled_count = int(np.floor(mix.controlled_fraction * n_vehicles))
    av_count = int(np.floor(mix.av_fraction * n_vehicles))
    bus_count = int(np.floor(mix.bus_fraction * n_vehicles))

    permutation = rng.permutation(n_vehicles)
    cursor = 0

    controlled_idx = permutation[cursor : cursor + controlled_count]
    cursor += controlled_count
    av_idx = permutation[cursor : cursor + av_count]
    cursor += av_count
    bus_idx = permutation[cursor : cursor + bus_count]

    veh_type[controlled_idx] = AV_VEH_TYPE
    behavior_id[controlled_idx] = CONTROLLED_AV_BEHAVIOR_ID
    controlled[controlled_idx] = True

    veh_type[av_idx] = AV_VEH_TYPE
    behavior_id[av_idx] = AV_BEHAVIOR_ID

    veh_type[bus_idx] = BUS_VEH_TYPE
    behavior_id[bus_idx] = BUS_BEHAVIOR_ID
    length[bus_idx] = mix.bus_length

    return veh_type, behavior_id, controlled, length


def make_uniform_random_state(
    *,
    num_lanes: int,
    road_length: int,
    density: float,
    seed: int | np.integer,
    vehicle_mix: VehicleMix | None = None,
) -> TrafficState:
    num_lanes = _validate_positive_int("num_lanes", num_lanes)
    road_length = _validate_positive_int("road_length", road_length)
    density = _validate_probability("density", density)
    _validate_seed(seed)

    if num_lanes > _MAX_LANE_VALUE + 1:
        raise ValueError(f"num_lanes must fit in {np.dtype(LANE_DTYPE)}, got {num_lanes}")

    mix = VehicleMix() if vehicle_mix is None else vehicle_mix

    rng = np.random.default_rng(seed)

    capacity = num_lanes * road_length
    n_vehicles = int(np.floor(capacity * density))
    veh_type, behavior_id, controlled, length = _assign_vehicle_types(n_vehicles, mix, rng)
    if int(length.astype(np.int64).sum()) > capacity:
        raise ValueError("could not place non-overlapping vehicle bodies for requested density/mix")
    lane = np.zeros(n_vehicles, dtype=LANE_DTYPE)
    pos = np.zeros(n_vehicles, dtype=POSITION_DTYPE)
    occ = np.zeros((num_lanes, road_length), dtype=bool)
    max_attempts = 16
    base_order = np.arange(n_vehicles, dtype=np.int64)
    order = np.argsort(-length.astype(np.int64), kind="stable")
    for _attempt in range(max_attempts):
        occ.fill(False)
        lane.fill(0)
        pos.fill(0)
        placed_all = True
        if _attempt > 0:
            order = order[rng.permutation(order.shape[0])]
        for idx in order:
            i = int(idx)
            l_i = int(length[i])
            candidates = rng.permutation(capacity)
            placed = False
            for flat in candidates:
                li = int(flat // road_length)
                pi = int(flat % road_length)
                cells = [((pi + d) % road_length) for d in range(l_i)]
                if any(occ[li, c] for c in cells):
                    continue
                lane[i] = li
                pos[i] = pi
                for c in cells:
                    occ[li, c] = True
                placed = True
                break
            if not placed:
                placed_all = False
                break
        if placed_all:
            break
    else:
        raise ValueError("could not place non-overlapping vehicle bodies for requested density/mix")

    return TrafficState(
        vehicle_id=np.arange(n_vehicles, dtype=VEHICLE_ID_DTYPE),
        lane=lane,
        pos=pos,
        vel=np.zeros(n_vehicles, dtype=VELOCITY_DTYPE),
        length=length,
        veh_type=veh_type,
        behavior_id=behavior_id,
        alive=np.ones(n_vehicles, dtype=BOOL_DTYPE),
        last_lane_delta=np.zeros(n_vehicles, dtype=LANE_DELTA_DTYPE),
        changed_lane=np.zeros(n_vehicles, dtype=BOOL_DTYPE),
        controlled=controlled,
    )
