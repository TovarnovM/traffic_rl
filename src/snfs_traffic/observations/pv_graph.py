"""PV-centred padded graph observations for centralized AV control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from snfs_traffic.control import _lateral_valid
from snfs_traffic.core import (
    SimulationParams,
    TrafficState,
    build_body_occupancy,
    build_lane_order,
    build_occupancy,
)
from snfs_traffic.scenarios import AV_VEH_TYPE, HDV_VEH_TYPE
from snfs_traffic.topology import RingTopology


PV_GRAPH_ACTION_VALUES = (-1, 0, 1)
PV_GRAPH_NEIGHBOR_RELATIONS = (
    "left_front",
    "left_back",
    "same_front",
    "same_back",
    "right_front",
    "right_back",
)

_BASE_NODE_FEATURES = (
    "pv_relative_position",
    "lane_norm",
    "pv_relative_lane",
    "speed_norm",
    "pv_relative_speed",
    "last_lane_delta",
    "cooldown_norm",
)
_LANE_NEIGHBOR_FEATURES = (
    "lane_exists",
    "front_present",
    "front_gap_norm",
    "front_relative_speed",
    "front_is_hdv",
    "front_is_pv",
    "back_present",
    "back_gap_norm",
    "back_relative_speed",
    "back_is_hdv",
    "back_is_pv",
)
PV_GRAPH_NODE_FEATURES = _BASE_NODE_FEATURES + tuple(
    f"{side}_{feature}"
    for side in ("left", "same", "right")
    for feature in _LANE_NEIGHBOR_FEATURES
) + (
    "nearest_hdv_front_present",
    "nearest_hdv_front_gap_norm",
    "nearest_hdv_front_relative_lane",
    "nearest_hdv_front_relative_speed",
    "nearest_hdv_back_present",
    "nearest_hdv_back_gap_norm",
    "nearest_hdv_back_relative_lane",
    "nearest_hdv_back_relative_speed",
)
PV_GRAPH_EDGE_FEATURES = (
    "relative_position",
    "relative_lane",
    "relative_speed",
)
PV_GRAPH_GLOBAL_FEATURES = (
    "pv_speed_norm",
    "pv_lane_norm",
    "density",
    "active_node_fraction",
)


@dataclass(frozen=True, slots=True)
class PvGraphConfig:
    """Geometry and feature limits for the first centralized graph MVP."""

    front_distance: int = 30
    back_distance: int = 10
    sensor_distance: int = 60
    cooldown_steps: int = 5

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("front_distance", self.front_distance, 0),
            ("back_distance", self.back_distance, 0),
            ("sensor_distance", self.sensor_distance, 1),
            ("cooldown_steps", self.cooldown_steps, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an int >= {minimum}")
        if self.front_distance + self.back_distance < 1:
            raise ValueError("the control window must contain at least one non-zero distance")

    def validate_for(self, params: SimulationParams) -> None:
        if self.front_distance + self.back_distance >= params.road_length:
            raise ValueError("front_distance + back_distance must be < road_length")
        if self.sensor_distance >= params.road_length:
            raise ValueError("sensor_distance must be < road_length")

    def max_nodes(self, params: SimulationParams) -> int:
        self.validate_for(params)
        return params.num_lanes * (self.front_distance + self.back_distance + 1)


@dataclass(frozen=True, slots=True)
class PvGraphObservation:
    """Fixed-size network input plus the private slot-to-vehicle mapping."""

    node_features: np.ndarray
    node_mask: np.ndarray
    neighbor_index: np.ndarray
    neighbor_mask: np.ndarray
    edge_features: np.ndarray
    action_mask: np.ndarray
    global_features: np.ndarray
    vehicle_id: np.ndarray

    @property
    def active_count(self) -> int:
        return int(np.sum(self.node_mask))

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return only arrays that belong to the public Gym observation."""

        return {
            "node_features": self.node_features,
            "node_mask": self.node_mask,
            "neighbor_index": self.neighbor_index,
            "neighbor_mask": self.neighbor_mask,
            "edge_features": self.edge_features,
            "action_mask": self.action_mask,
            "global_features": self.global_features,
        }


def _signed_pv_distance(position: int, pv_position: int, road_length: int, config: PvGraphConfig) -> int | None:
    forward = (position - pv_position) % road_length
    if forward <= config.front_distance:
        return int(forward)
    behind = (pv_position - position) % road_length
    if behind <= config.back_distance:
        return -int(behind)
    return None


def _nearest_by_direction(
    state: TrafficState,
    *,
    ego_idx: int,
    candidate_indices: np.ndarray,
    direction: str,
    road_length: int,
) -> tuple[int | None, int | None]:
    if candidate_indices.size == 0:
        return None, None
    ego_pos = int(state.pos[ego_idx])
    if direction == "front":
        distances = (state.pos[candidate_indices].astype(np.int64) - ego_pos) % road_length
    elif direction == "back":
        distances = (ego_pos - state.pos[candidate_indices].astype(np.int64)) % road_length
    else:  # pragma: no cover - private caller contract
        raise ValueError("direction must be 'front' or 'back'")
    valid = candidate_indices != ego_idx
    if not np.any(valid):
        return None, None
    candidates = candidate_indices[valid]
    distances = distances[valid]
    order = np.lexsort((state.vehicle_id[candidates], distances))
    selected = int(candidates[int(order[0])])
    distance = int(distances[int(order[0])])
    return selected, distance


def _type_flags(state: TrafficState, idx: int, pv_idx: int) -> tuple[float, float]:
    return (
        1.0 if int(state.veh_type[idx]) == HDV_VEH_TYPE else 0.0,
        1.0 if idx == pv_idx else 0.0,
    )


def build_pv_graph_observation(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    *,
    priority_vehicle_id: int,
    config: PvGraphConfig | None = None,
    cooldown_by_vehicle_id: Mapping[int, int] | None = None,
) -> PvGraphObservation:
    """Build a sparse fixed-neighbour graph for AVs near one PV.

    Graph slots are rebuilt every step and sorted by PV-relative position,
    lane, and stable physical ``vehicle_id``.  ``vehicle_id`` is deliberately
    not exposed to the network; it is retained only for applying this step's
    actions to the simulator.
    """

    cfg = config or PvGraphConfig()
    cfg.validate_for(params)
    if topology.boundary != "periodic":
        raise ValueError("topology must be periodic")
    if topology.num_lanes != params.num_lanes or topology.length != params.road_length:
        raise ValueError("topology must match params")
    priority_matches = np.flatnonzero(
        state.alive & (state.vehicle_id == int(priority_vehicle_id))
    )
    if priority_matches.size != 1:
        raise ValueError("priority_vehicle_id must identify exactly one alive vehicle")
    pv_idx = int(priority_matches[0])
    pv_pos = int(state.pos[pv_idx])
    cooldowns = cooldown_by_vehicle_id or {}
    max_nodes = cfg.max_nodes(params)
    max_speed = max(params.vmax_default, params.vmax_controlled, 1)
    lane_scale = max(params.num_lanes - 1, 1)

    candidates: list[tuple[int, int, int, int]] = []
    for raw_idx in np.flatnonzero(
        state.alive & (state.veh_type == AV_VEH_TYPE)
    ):
        idx = int(raw_idx)
        signed_distance = _signed_pv_distance(
            int(state.pos[idx]), pv_pos, params.road_length, cfg
        )
        if signed_distance is None:
            continue
        candidates.append(
            (
                signed_distance,
                int(state.lane[idx]),
                int(state.vehicle_id[idx]),
                idx,
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(candidates) > max_nodes:
        raise RuntimeError(
            f"control window contains {len(candidates)} AVs but max_nodes={max_nodes}"
        )

    node_features = np.zeros(
        (max_nodes, len(PV_GRAPH_NODE_FEATURES)), dtype=np.float32
    )
    node_mask = np.zeros(max_nodes, dtype=np.int8)
    neighbor_index = np.zeros(
        (max_nodes, len(PV_GRAPH_NEIGHBOR_RELATIONS)), dtype=np.int32
    )
    neighbor_mask = np.zeros_like(neighbor_index, dtype=np.int8)
    edge_features = np.zeros(
        (
            max_nodes,
            len(PV_GRAPH_NEIGHBOR_RELATIONS),
            len(PV_GRAPH_EDGE_FEATURES),
        ),
        dtype=np.float32,
    )
    action_mask = np.zeros((max_nodes, 3), dtype=np.int8)
    action_mask[:, 1] = 1  # Padding and active nodes always have a valid stay.
    vehicle_ids = np.full(max_nodes, -1, dtype=np.int32)

    active_indices = np.asarray([item[3] for item in candidates], dtype=np.int64)
    signed_distances = np.asarray([item[0] for item in candidates], dtype=np.int32)
    slot_by_state_idx = {int(idx): slot for slot, idx in enumerate(active_indices)}
    alive_indices = np.flatnonzero(state.alive).astype(np.int64, copy=False)
    hdv_indices = np.flatnonzero(
        state.alive & (state.veh_type == HDV_VEH_TYPE)
    ).astype(np.int64, copy=False)

    occupancy = build_occupancy(state, params)
    body_occupancy = build_body_occupancy(state, params)
    lane_order, lane_counts, _lane_rank = build_lane_order(
        occupancy, n_vehicles=state.n_vehicles
    )

    for slot, idx_raw in enumerate(active_indices):
        idx = int(idx_raw)
        vehicle_id = int(state.vehicle_id[idx])
        lane = int(state.lane[idx])
        speed = int(state.vel[idx])
        signed_distance = int(signed_distances[slot])
        node_mask[slot] = 1
        vehicle_ids[slot] = np.int32(vehicle_id)
        if signed_distance >= 0:
            node_features[slot, 0] = signed_distance / max(cfg.front_distance, 1)
        else:
            node_features[slot, 0] = signed_distance / max(cfg.back_distance, 1)
        node_features[slot, 1] = lane / lane_scale if params.num_lanes > 1 else 0.0
        node_features[slot, 2] = (lane - int(state.lane[pv_idx])) / lane_scale
        node_features[slot, 3] = speed / max_speed
        node_features[slot, 4] = (speed - int(state.vel[pv_idx])) / max_speed
        node_features[slot, 5] = float(state.last_lane_delta[idx])
        remaining_cooldown = max(0, int(cooldowns.get(vehicle_id, 0)))
        node_features[slot, 6] = remaining_cooldown / max(cfg.cooldown_steps, 1)

        feature_cursor = len(_BASE_NODE_FEATURES)
        for lane_delta in (-1, 0, 1):
            target_lane = lane + lane_delta
            if not 0 <= target_lane < params.num_lanes:
                feature_cursor += len(_LANE_NEIGHBOR_FEATURES)
                continue
            node_features[slot, feature_cursor] = 1.0
            lane_candidates = alive_indices[state.lane[alive_indices] == target_lane]
            front_idx, front_distance = _nearest_by_direction(
                state,
                ego_idx=idx,
                candidate_indices=lane_candidates,
                direction="front",
                road_length=params.road_length,
            )
            back_idx, back_distance = _nearest_by_direction(
                state,
                ego_idx=idx,
                candidate_indices=lane_candidates,
                direction="back",
                road_length=params.road_length,
            )
            if front_idx is not None and front_distance is not None and front_distance <= cfg.sensor_distance:
                is_hdv, is_pv = _type_flags(state, front_idx, pv_idx)
                node_features[slot, feature_cursor + 1] = 1.0
                node_features[slot, feature_cursor + 2] = front_distance / cfg.sensor_distance
                node_features[slot, feature_cursor + 3] = (
                    int(state.vel[front_idx]) - speed
                ) / max_speed
                node_features[slot, feature_cursor + 4] = is_hdv
                node_features[slot, feature_cursor + 5] = is_pv
            if back_idx is not None and back_distance is not None and back_distance <= cfg.sensor_distance:
                is_hdv, is_pv = _type_flags(state, back_idx, pv_idx)
                node_features[slot, feature_cursor + 6] = 1.0
                node_features[slot, feature_cursor + 7] = back_distance / cfg.sensor_distance
                node_features[slot, feature_cursor + 8] = (
                    int(state.vel[back_idx]) - speed
                ) / max_speed
                node_features[slot, feature_cursor + 9] = is_hdv
                node_features[slot, feature_cursor + 10] = is_pv
            feature_cursor += len(_LANE_NEIGHBOR_FEATURES)

        hdv_feature_cursor = len(_BASE_NODE_FEATURES) + 3 * len(
            _LANE_NEIGHBOR_FEATURES
        )
        for direction, offset in (("front", 0), ("back", 4)):
            hdv_idx, hdv_distance = _nearest_by_direction(
                state,
                ego_idx=idx,
                candidate_indices=hdv_indices,
                direction=direction,
                road_length=params.road_length,
            )
            if hdv_idx is None or hdv_distance is None or hdv_distance > cfg.sensor_distance:
                continue
            node_features[slot, hdv_feature_cursor + offset] = 1.0
            node_features[slot, hdv_feature_cursor + offset + 1] = (
                hdv_distance / cfg.sensor_distance
            )
            node_features[slot, hdv_feature_cursor + offset + 2] = (
                int(state.lane[hdv_idx]) - lane
            ) / lane_scale
            node_features[slot, hdv_feature_cursor + offset + 3] = (
                int(state.vel[hdv_idx]) - speed
            ) / max_speed

        if remaining_cooldown == 0:
            for action_col, lane_delta in ((0, -1), (2, 1)):
                target_lane = lane + lane_delta
                if not 0 <= target_lane < params.num_lanes:
                    continue
                valid, _reason = _lateral_valid(
                    state,
                    lane_order,
                    lane_counts,
                    occupancy,
                    body_occupancy,
                    idx,
                    target_lane,
                    params.road_length,
                )
                action_mask[slot, action_col] = int(valid)

    relation_slot = 0
    for lane_delta in (-1, 0, 1):
        for direction in ("front", "back"):
            for source_slot, source_idx_raw in enumerate(active_indices):
                source_idx = int(source_idx_raw)
                target_lane = int(state.lane[source_idx]) + lane_delta
                if not 0 <= target_lane < params.num_lanes:
                    continue
                active_lane = active_indices[state.lane[active_indices] == target_lane]
                neighbor_idx, _distance = _nearest_by_direction(
                    state,
                    ego_idx=source_idx,
                    candidate_indices=active_lane,
                    direction=direction,
                    road_length=params.road_length,
                )
                if neighbor_idx is None:
                    continue
                target_slot = slot_by_state_idx[int(neighbor_idx)]
                neighbor_index[source_slot, relation_slot] = np.int32(target_slot)
                neighbor_mask[source_slot, relation_slot] = 1
                delta_position = int(signed_distances[target_slot]) - int(
                    signed_distances[source_slot]
                )
                edge_features[source_slot, relation_slot, 0] = np.clip(
                    delta_position / max(cfg.front_distance + cfg.back_distance, 1),
                    -1.0,
                    1.0,
                )
                edge_features[source_slot, relation_slot, 1] = (
                    int(state.lane[neighbor_idx]) - int(state.lane[source_idx])
                ) / lane_scale
                edge_features[source_slot, relation_slot, 2] = (
                    int(state.vel[neighbor_idx]) - int(state.vel[source_idx])
                ) / max_speed
            relation_slot += 1

    alive_count = int(np.sum(state.alive))
    global_features = np.asarray(
        [
            int(state.vel[pv_idx]) / max_speed,
            int(state.lane[pv_idx]) / lane_scale if params.num_lanes > 1 else 0.0,
            alive_count / (params.num_lanes * params.road_length),
            len(candidates) / max_nodes,
        ],
        dtype=np.float32,
    )

    return PvGraphObservation(
        node_features=np.ascontiguousarray(node_features),
        node_mask=np.ascontiguousarray(node_mask),
        neighbor_index=np.ascontiguousarray(neighbor_index),
        neighbor_mask=np.ascontiguousarray(neighbor_mask),
        edge_features=np.ascontiguousarray(edge_features),
        action_mask=np.ascontiguousarray(action_mask),
        global_features=np.ascontiguousarray(global_features),
        vehicle_id=np.ascontiguousarray(vehicle_ids),
    )
