"""Metrics for priority-vehicle baseline experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from snfs_traffic.control import LateralOverrideResult
from snfs_traffic.core import SimulationParams, TrafficState, priority_vehicle_mask

if TYPE_CHECKING:  # pragma: no cover
    from snfs_traffic.rules import PriorityYieldDecision


@dataclass(frozen=True, slots=True)
class PriorityMetricsSummary:
    measurement_steps: int
    vehicle_count: int
    priority_vehicle_count: int
    density: float
    flow_all: float
    flow_background: float
    flow_priority: float
    mean_speed_all: float
    mean_speed_background: float
    mean_speed_priority: float
    worst_priority_mean_speed: float
    mean_priority_time_loss: float
    worst_priority_time_loss: float
    stopped_fraction_all: float
    stopped_fraction_background: float
    stopped_fraction_priority: float
    lane_change_rate_all: float
    lane_change_rate_background: float
    lane_change_rate_priority: float
    completed_priority_laps: int
    mean_priority_lap_time: float | None
    rule_detections: int
    rule_requests: int
    rule_applied: int
    rule_rejected: int
    rule_blocked_no_safe_lane: int
    rule_cooldown_suppressed: int
    per_priority_vehicle: dict[str, dict[str, float | int | None]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PriorityMetricsAccumulator:
    """Accumulate step-level measurements without retaining trajectories."""

    def __init__(
        self,
        params: SimulationParams,
        *,
        priority_vehicle_ids: tuple[int, ...] | None = None,
    ) -> None:
        self.params = params
        self._configured_priority_ids = priority_vehicle_ids
        self._priority_ids: tuple[int, ...] | None = None
        self._steps = 0
        self._vehicle_count = 0
        self._sum_density = 0.0
        self._sum_flow_all = 0.0
        self._sum_flow_background = 0.0
        self._sum_flow_priority = 0.0
        self._sum_speed_all = 0.0
        self._sum_speed_background = 0.0
        self._sum_speed_priority = 0.0
        self._sum_stopped_all = 0.0
        self._sum_stopped_background = 0.0
        self._sum_stopped_priority = 0.0
        self._lane_changes_all = 0
        self._lane_changes_background = 0
        self._lane_changes_priority = 0
        self._priority_speed_sum: dict[int, float] = {}
        self._priority_distance: dict[int, float] = {}
        self._priority_laps: dict[int, int] = {}
        self._priority_last_lap_step: dict[int, int] = {}
        self._priority_lap_times: dict[int, list[int]] = {}
        self._rule_detections = 0
        self._rule_requests = 0
        self._rule_applied = 0
        self._rule_rejected = 0
        self._rule_blocked = 0
        self._rule_cooldown = 0

    def _initialize_priority(self, state: TrafficState) -> None:
        if self._configured_priority_ids is None:
            ids = tuple(
                int(value)
                for value in state.vehicle_id[
                    state.alive & priority_vehicle_mask(state)
                ]
            )
        else:
            ids = tuple(int(value) for value in self._configured_priority_ids)
        if not ids:
            raise ValueError("priority metrics require at least one priority vehicle")
        if len(set(ids)) != len(ids):
            raise ValueError("priority_vehicle_ids must be unique")
        alive_ids = {int(value) for value in state.vehicle_id[state.alive]}
        missing = [vehicle_id for vehicle_id in ids if vehicle_id not in alive_ids]
        if missing:
            raise ValueError(f"priority vehicles are not alive: {missing}")
        self._priority_ids = ids
        self._priority_speed_sum = {vehicle_id: 0.0 for vehicle_id in ids}
        self._priority_distance = {vehicle_id: 0.0 for vehicle_id in ids}
        self._priority_laps = {vehicle_id: 0 for vehicle_id in ids}
        self._priority_last_lap_step = {vehicle_id: 0 for vehicle_id in ids}
        self._priority_lap_times = {vehicle_id: [] for vehicle_id in ids}

    def observe(
        self,
        state: TrafficState,
        *,
        decision: "PriorityYieldDecision | None" = None,
        result: LateralOverrideResult | None = None,
    ) -> None:
        if self._priority_ids is None:
            self._initialize_priority(state)
        assert self._priority_ids is not None
        id_to_idx = {
            int(vehicle_id): idx for idx, vehicle_id in enumerate(state.vehicle_id)
        }
        pv_indices = np.asarray(
            [id_to_idx[vehicle_id] for vehicle_id in self._priority_ids], dtype=np.int64
        )
        if not np.all(state.alive[pv_indices]):
            raise ValueError("a configured priority vehicle is no longer alive")
        alive = state.alive
        pv = np.zeros(state.n_vehicles, dtype=np.bool_)
        pv[pv_indices] = True
        background = alive & ~pv
        alive_count = int(np.sum(alive))
        background_count = int(np.sum(background))
        capacity = self.params.num_lanes * self.params.road_length
        self._steps += 1
        self._vehicle_count = alive_count
        self._sum_density += alive_count / capacity
        self._sum_flow_all += (
            float(np.sum(state.vel[alive], dtype=np.float64)) / capacity
        )
        self._sum_flow_background += (
            float(np.sum(state.vel[background], dtype=np.float64)) / capacity
        )
        self._sum_flow_priority += (
            float(np.sum(state.vel[pv], dtype=np.float64)) / capacity
        )
        self._sum_speed_all += float(np.mean(state.vel[alive])) if alive_count else 0.0
        self._sum_speed_background += (
            float(np.mean(state.vel[background])) if background_count else 0.0
        )
        self._sum_speed_priority += float(np.mean(state.vel[pv]))
        self._sum_stopped_all += (
            float(np.mean(state.vel[alive] == 0)) if alive_count else 0.0
        )
        self._sum_stopped_background += (
            float(np.mean(state.vel[background] == 0)) if background_count else 0.0
        )
        self._sum_stopped_priority += float(np.mean(state.vel[pv] == 0))
        self._lane_changes_all += int(np.sum(state.changed_lane[alive]))
        self._lane_changes_background += int(np.sum(state.changed_lane[background]))
        self._lane_changes_priority += int(np.sum(state.changed_lane[pv]))

        for vehicle_id, idx in zip(self._priority_ids, pv_indices):
            speed = float(state.vel[idx])
            previous_distance = self._priority_distance[vehicle_id]
            new_distance = previous_distance + speed
            previous_laps = int(previous_distance // self.params.road_length)
            new_laps = int(new_distance // self.params.road_length)
            self._priority_speed_sum[vehicle_id] += speed
            self._priority_distance[vehicle_id] = new_distance
            for _lap in range(previous_laps, new_laps):
                lap_time = self._steps - self._priority_last_lap_step[vehicle_id]
                self._priority_lap_times[vehicle_id].append(lap_time)
                self._priority_last_lap_step[vehicle_id] = self._steps
            self._priority_laps[vehicle_id] = new_laps

        if decision is not None:
            self._rule_detections += len(decision.detected_vehicle_ids)
            self._rule_requests += decision.request_count
            self._rule_blocked += len(decision.blocked_vehicle_ids)
            self._rule_cooldown += len(decision.cooldown_vehicle_ids)
        if result is not None:
            applied = int(np.sum(result.applied))
            self._rule_applied += applied
            self._rule_rejected += int(result.vehicle_id.size) - applied

    def summary(self) -> PriorityMetricsSummary:
        if self._steps == 0 or self._priority_ids is None:
            raise ValueError("cannot summarize zero measurement steps")
        steps = self._steps
        pv_count = len(self._priority_ids)
        background_count = self._vehicle_count - pv_count
        per_priority: dict[str, dict[str, float | int | None]] = {}
        all_lap_times: list[int] = []
        per_pv_speeds: list[float] = []
        per_pv_losses: list[float] = []
        for vehicle_id in self._priority_ids:
            mean_speed = self._priority_speed_sum[vehicle_id] / steps
            time_loss = 1.0 - mean_speed / max(1, self.params.vmax_controlled)
            lap_times = self._priority_lap_times[vehicle_id]
            all_lap_times.extend(lap_times)
            per_pv_speeds.append(mean_speed)
            per_pv_losses.append(time_loss)
            per_priority[str(vehicle_id)] = {
                "mean_speed": mean_speed,
                "time_loss": time_loss,
                "distance": self._priority_distance[vehicle_id],
                "completed_laps": self._priority_laps[vehicle_id],
                "mean_lap_time": float(np.mean(lap_times)) if lap_times else None,
            }
        return PriorityMetricsSummary(
            measurement_steps=steps,
            vehicle_count=self._vehicle_count,
            priority_vehicle_count=pv_count,
            density=self._sum_density / steps,
            flow_all=self._sum_flow_all / steps,
            flow_background=self._sum_flow_background / steps,
            flow_priority=self._sum_flow_priority / steps,
            mean_speed_all=self._sum_speed_all / steps,
            mean_speed_background=self._sum_speed_background / steps,
            mean_speed_priority=self._sum_speed_priority / steps,
            worst_priority_mean_speed=min(per_pv_speeds),
            mean_priority_time_loss=float(np.mean(per_pv_losses)),
            worst_priority_time_loss=max(per_pv_losses),
            stopped_fraction_all=self._sum_stopped_all / steps,
            stopped_fraction_background=self._sum_stopped_background / steps,
            stopped_fraction_priority=self._sum_stopped_priority / steps,
            lane_change_rate_all=self._lane_changes_all
            / (steps * max(1, self._vehicle_count)),
            lane_change_rate_background=self._lane_changes_background
            / (steps * max(1, background_count)),
            lane_change_rate_priority=self._lane_changes_priority / (steps * pv_count),
            completed_priority_laps=sum(self._priority_laps.values()),
            mean_priority_lap_time=float(np.mean(all_lap_times))
            if all_lap_times
            else None,
            rule_detections=self._rule_detections,
            rule_requests=self._rule_requests,
            rule_applied=self._rule_applied,
            rule_rejected=self._rule_rejected,
            rule_blocked_no_safe_lane=self._rule_blocked,
            rule_cooldown_suppressed=self._rule_cooldown,
            per_priority_vehicle=per_priority,
        )


def paired_bootstrap_interval(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of paired differences."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a non-empty finite 1D array")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be an int >= 1")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = np.mean(values[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)
