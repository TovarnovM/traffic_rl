from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snfs_traffic.control import compute_lateral_action_mask, controlled_vehicle_ids
from snfs_traffic.core import SimulationParams, TrafficState
from snfs_traffic.core.indexing import MISSING_INDEX, build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.lane_change_kernels import target_lane_neighbors_at_pos_kernel
from snfs_traffic.core.state import validate_state
from snfs_traffic.topology import RingTopology

LOCAL_ACTION_VALUES = (-1, 0, 1)
LOCAL_OBSERVATION_FEATURES = (
    "ego_lane_norm", "ego_pos_norm", "ego_vel_norm", "front_gap_norm", "front_rel_speed_norm", "back_gap_norm", "back_rel_speed_norm",
    "left_exists", "left_cell_free", "left_front_gap_norm", "left_front_rel_speed_norm", "left_back_gap_norm", "left_back_rel_speed_norm", "left_safe",
    "right_exists", "right_cell_free", "right_front_gap_norm", "right_front_rel_speed_norm", "right_back_gap_norm", "right_back_rel_speed_norm", "right_safe",
)

@dataclass(frozen=True, slots=True)
class LocalObservationConfig:
    include_position: bool = True
    dtype: np.dtype | type = np.float32

@dataclass(frozen=True, slots=True)
class LocalObservationBatch:
    vehicle_id: np.ndarray
    obs: np.ndarray
    action_mask: np.ndarray
    feature_names: tuple[str, ...]
    action_values: tuple[int, int, int] = LOCAL_ACTION_VALUES


def build_local_observations(state: TrafficState, params: SimulationParams, topology: RingTopology, config: LocalObservationConfig | None = None) -> LocalObservationBatch:
    cfg = config or LocalObservationConfig()
    validate_state(state, params)
    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, back_id, front_gap, back_gap = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)
    vids, mask = compute_lateral_action_mask(state, params, topology)
    vid_to_idx = {int(v): i for i, v in enumerate(state.vehicle_id)}
    denom_lane = max(params.num_lanes - 1, 1)
    denom_speed = max(params.vmax_default, params.vmax_controlled, 1)
    denom_gap = max(params.road_length - 1, 1)
    obs = np.zeros((vids.shape[0], len(LOCAL_OBSERVATION_FEATURES)), dtype=cfg.dtype)
    for r, vid in enumerate(vids):
        i = vid_to_idx[int(vid)]
        ego_lane = int(state.lane[i]); ego_pos = int(state.pos[i]); ego_vel = int(state.vel[i])
        obs[r, 0] = ego_lane / denom_lane if params.num_lanes > 1 else 0.0
        obs[r, 1] = (ego_pos / params.road_length) if cfg.include_position else 0.0
        obs[r, 2] = ego_vel / denom_speed

        def neighbor_feats(nid, gap):
            if int(nid) == MISSING_INDEX or int(nid) == i:
                return (params.road_length - 1, 0.0)
            return (int(gap), (int(state.vel[int(nid)]) - ego_vel) / denom_speed)

        g, rs = neighbor_feats(front_id[i], front_gap[i]); obs[r, 3] = g / denom_gap; obs[r, 4] = rs
        g, rs = neighbor_feats(back_id[i], back_gap[i]); obs[r, 5] = g / denom_gap; obs[r, 6] = rs

        for base, delta in [(7, -1), (14, 1)]:
            tl = ego_lane + delta
            if not (0 <= tl < params.num_lanes):
                continue
            obs[r, base] = 1.0
            cell_free = int(occupancy[tl, ego_pos]) == MISSING_INDEX
            obs[r, base + 1] = 1.0 if cell_free else 0.0
            if int(lane_counts[tl]) == 0:
                obs[r, base + 2] = (params.road_length - 1) / denom_gap
                obs[r, base + 4] = (params.road_length - 1) / denom_gap
                obs[r, base + 6] = 1.0 if cell_free else 0.0
            else:
                tf, tb, gnf, gnb = target_lane_neighbors_at_pos_kernel(state.pos, lane_order, lane_counts, target_lane=tl, candidate_pos=ego_pos, road_length=params.road_length)
                obs[r, base + 2] = gnf / denom_gap
                obs[r, base + 3] = (int(state.vel[int(tf)]) - ego_vel) / denom_speed
                obs[r, base + 4] = gnb / denom_gap
                obs[r, base + 5] = (int(state.vel[int(tb)]) - ego_vel) / denom_speed
                obs[r, base + 6] = 1.0 if (cell_free and (ego_vel > (int(state.vel[int(tb)]) - int(gnb)))) else 0.0

    return LocalObservationBatch(vehicle_id=vids, obs=obs, action_mask=mask, feature_names=LOCAL_OBSERVATION_FEATURES)
