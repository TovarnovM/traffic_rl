"""Deterministic Baseline-2 rule: yield when a priority vehicle approaches."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from snfs_traffic.control import (
    LANE_LEFT,
    LANE_RIGHT,
    LateralOverrideResult,
    _lateral_valid,
)
from snfs_traffic.core import (
    SimulationParams,
    TrafficState,
    build_body_occupancy,
    build_lane_order,
    build_occupancy,
    priority_vehicle_mask,
)
from snfs_traffic.core.lane_change_kernels import target_lane_neighbors_at_pos_kernel
from snfs_traffic.core.indexing import MISSING_INDEX
from snfs_traffic.topology import RingTopology


@dataclass(frozen=True, slots=True)
class PriorityYieldConfig:
    detection_distance: int = 30
    cooldown_steps: int = 5

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("detection_distance", self.detection_distance, 1),
            ("cooldown_steps", self.cooldown_steps, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an int >= {minimum}")


@dataclass(frozen=True, slots=True)
class PriorityYieldDecision:
    overrides: Mapping[int, int]
    detected_vehicle_ids: tuple[int, ...]
    blocked_vehicle_ids: tuple[int, ...]
    cooldown_vehicle_ids: tuple[int, ...]

    @property
    def request_count(self) -> int:
        return len(self.overrides)


class PriorityYieldController:
    """Stateful yield-rule controller with a per-vehicle lane-change cooldown."""

    def __init__(
        self,
        config: PriorityYieldConfig | None = None,
        *,
        eligible_vehicle_ids: Iterable[int] | None = None,
    ) -> None:
        self.config = config or PriorityYieldConfig()
        if eligible_vehicle_ids is None:
            self._eligible_ids = None
        else:
            values = tuple(eligible_vehicle_ids)
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in values
            ):
                raise ValueError("eligible_vehicle_ids must contain integers")
            if len(set(int(value) for value in values)) != len(values):
                raise ValueError("eligible_vehicle_ids must be unique")
            self._eligible_ids = frozenset(int(value) for value in values)
        self._cooldown_by_id: dict[int, int] = {}

    def reset(self) -> None:
        self._cooldown_by_id.clear()

    def _tick_cooldowns(self) -> None:
        expired: list[int] = []
        for vehicle_id, remaining in list(self._cooldown_by_id.items()):
            next_remaining = remaining - 1
            if next_remaining <= 0:
                expired.append(vehicle_id)
            else:
                self._cooldown_by_id[vehicle_id] = next_remaining
        for vehicle_id in expired:
            del self._cooldown_by_id[vehicle_id]

    def decide(
        self,
        state: TrafficState,
        params: SimulationParams,
        topology: RingTopology,
    ) -> PriorityYieldDecision:
        if (
            topology.num_lanes != params.num_lanes
            or topology.length != params.road_length
        ):
            raise ValueError("topology must match params")
        self._tick_cooldowns()
        priority = state.alive & priority_vehicle_mask(state)
        priority_indices = np.flatnonzero(priority)
        if priority_indices.size == 0:
            return PriorityYieldDecision(MappingProxyType({}), (), (), ())

        overrides: dict[int, int] = {}
        detected: list[int] = []
        blocked: list[int] = []
        cooling: list[int] = []
        candidates: list[tuple[int, int, int]] = []
        priority_by_lane = {
            lane: priority_indices[state.lane[priority_indices] == lane]
            for lane in range(params.num_lanes)
        }

        order = np.argsort(state.vehicle_id, kind="stable")
        for idx_raw in order:
            idx = int(idx_raw)
            if not bool(state.alive[idx]) or bool(priority[idx]):
                continue
            vehicle_id = int(state.vehicle_id[idx])
            if self._eligible_ids is not None and vehicle_id not in self._eligible_ids:
                continue
            lane = int(state.lane[idx])
            lane_priority = priority_by_lane[lane]
            if lane_priority.size == 0:
                continue
            nearest_behind = min(
                int((int(state.pos[idx]) - int(state.pos[pv_idx])) % params.road_length)
                for pv_idx in lane_priority
            )
            if nearest_behind <= 0 or nearest_behind > self.config.detection_distance:
                continue
            detected.append(vehicle_id)
            if vehicle_id in self._cooldown_by_id:
                cooling.append(vehicle_id)
                continue
            candidates.append((idx, vehicle_id, lane))

        if not candidates:
            return PriorityYieldDecision(
                MappingProxyType(overrides),
                tuple(detected),
                tuple(blocked),
                tuple(cooling),
            )

        occupancy = build_occupancy(state, params)
        body_occupancy = build_body_occupancy(state, params)
        lane_order, lane_counts, _lane_rank = build_lane_order(
            occupancy, n_vehicles=state.n_vehicles
        )

        for idx, vehicle_id, lane in candidates:
            targets: list[tuple[int, int]] = []
            for delta in (LANE_LEFT, LANE_RIGHT):
                target_lane = lane + delta
                if target_lane < 0 or target_lane >= params.num_lanes:
                    continue
                safe, _reason = _lateral_valid(
                    state,
                    lane_order,
                    lane_counts,
                    occupancy,
                    body_occupancy,
                    idx,
                    target_lane,
                    params.road_length,
                )
                if not safe:
                    continue
                front, back, front_gap, back_gap = target_lane_neighbors_at_pos_kernel(
                    state.pos,
                    lane_order,
                    lane_counts,
                    target_lane=target_lane,
                    candidate_pos=int(state.pos[idx]),
                    road_length=params.road_length,
                )
                if front == MISSING_INDEX:
                    clearance = params.road_length - int(state.length[idx])
                else:
                    front_clearance = int(front_gap) - int(state.length[idx]) + 1
                    back_clearance = int(back_gap) - int(state.length[back]) + 1
                    clearance = min(front_clearance, back_clearance)
                targets.append((clearance, delta))

            if not targets:
                blocked.append(vehicle_id)
                continue
            best_clearance = max(item[0] for item in targets)
            best_deltas = sorted(
                delta for clearance, delta in targets if clearance == best_clearance
            )
            overrides[vehicle_id] = best_deltas[vehicle_id % len(best_deltas)]

        return PriorityYieldDecision(
            MappingProxyType(overrides),
            tuple(detected),
            tuple(blocked),
            tuple(cooling),
        )

    def observe(self, result: LateralOverrideResult | None) -> None:
        """Start cooldowns only for requests that were actually applied."""

        if result is None or self.config.cooldown_steps == 0:
            return
        for vehicle_id, applied in zip(result.vehicle_id, result.applied):
            if bool(applied):
                # ``decide`` ticks before checking, hence +1 gives exactly the
                # configured number of suppressed future decisions.
                self._cooldown_by_id[int(vehicle_id)] = self.config.cooldown_steps + 1
