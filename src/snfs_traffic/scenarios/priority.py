"""Priority-vehicle placement and fair Baseline-2 participant selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snfs_traffic.core import PRIORITY_VEH_TYPE, TrafficState, priority_vehicle_mask

from .init import (
    AV_VEH_TYPE,
    PRIORITY_BEHAVIOR_ID,
)


PRIORITY_YIELD_BEHAVIOR_ID = 5


@dataclass(frozen=True, slots=True)
class PriorityVehicleConfig:
    """Configuration for tagging vehicles in an already stabilized snapshot.

    ``target_gap`` is the desired number of empty cells from the rear PV body
    to the next PV head.  It is used only for ``placement="convoy"``.
    """

    count: int = 1
    placement: str = "random"
    target_gap: int = 15
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 1
        ):
            raise ValueError("count must be an int >= 1")
        if self.placement not in {"random", "convoy"}:
            raise ValueError("placement must be 'random' or 'convoy'")
        if (
            isinstance(self.target_gap, bool)
            or not isinstance(self.target_gap, int)
            or self.target_gap < 0
        ):
            raise ValueError("target_gap must be an int >= 0")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an int")


@dataclass(frozen=True, slots=True)
class PriorityVehicleAssignment:
    state: TrafficState
    vehicle_ids: tuple[int, ...]
    lane: int | None
    actual_gaps: tuple[int, ...]


def _eligible_unit_vehicle_indices(state: TrafficState) -> np.ndarray:
    return np.flatnonzero(
        state.alive & (state.length == 1) & ~priority_vehicle_mask(state)
    )


def _convoy_candidates(
    state: TrafficState,
    *,
    count: int,
    target_gap: int,
    road_length: int,
) -> list[tuple[int, tuple[int, ...], tuple[int, ...], int]]:
    candidates: list[tuple[int, tuple[int, ...], tuple[int, ...], int]] = []
    eligible = _eligible_unit_vehicle_indices(state)
    for lane in sorted(int(value) for value in np.unique(state.lane[eligible])):
        lane_indices = eligible[state.lane[eligible] == lane]
        ordered = sorted(
            (int(idx) for idx in lane_indices),
            key=lambda idx: (int(state.pos[idx]), int(state.vehicle_id[idx])),
        )
        if len(ordered) < count:
            continue
        for start in range(len(ordered)):
            selected = tuple(
                ordered[(start + offset) % len(ordered)] for offset in range(count)
            )
            gaps = tuple(
                int(
                    (
                        int(state.pos[selected[offset + 1]])
                        - int(state.pos[selected[offset]])
                        - int(state.length[selected[offset]])
                    )
                    % road_length
                )
                for offset in range(count - 1)
            )
            score = sum(abs(gap - target_gap) for gap in gaps)
            candidates.append((score, selected, gaps, lane))
    return candidates


def assign_priority_vehicles(
    state: TrafficState,
    config: PriorityVehicleConfig,
    *,
    road_length: int,
) -> PriorityVehicleAssignment:
    """Tag priority vehicles without changing positions, speeds, or density."""

    if (
        isinstance(road_length, bool)
        or not isinstance(road_length, int)
        or road_length < 1
    ):
        raise ValueError("road_length must be an int >= 1")
    eligible = _eligible_unit_vehicle_indices(state)
    if eligible.size < config.count:
        raise ValueError(
            f"need {config.count} alive unit-length vehicles to assign priority roles"
        )

    rng = np.random.default_rng(config.seed)
    lane: int | None = None
    gaps: tuple[int, ...] = ()
    if config.placement == "random" or config.count == 1:
        chosen = tuple(
            int(idx) for idx in rng.choice(eligible, size=config.count, replace=False)
        )
        chosen = tuple(sorted(chosen, key=lambda idx: int(state.vehicle_id[idx])))
        if config.count == 1:
            lane = int(state.lane[chosen[0]])
    else:
        candidates = _convoy_candidates(
            state,
            count=config.count,
            target_gap=config.target_gap,
            road_length=road_length,
        )
        if not candidates:
            raise ValueError(
                f"no lane contains {config.count} eligible vehicles for convoy placement"
            )
        best_score = min(candidate[0] for candidate in candidates)
        best = [candidate for candidate in candidates if candidate[0] == best_score]
        _score, chosen, gaps, lane = best[int(rng.integers(0, len(best)))]

    out = state.copy()
    for idx in chosen:
        out.veh_type[idx] = np.asarray(PRIORITY_VEH_TYPE, dtype=out.veh_type.dtype)
        out.behavior_id[idx] = np.asarray(
            PRIORITY_BEHAVIOR_ID, dtype=out.behavior_id.dtype
        )
        out.controlled[idx] = False

    vehicle_ids = tuple(int(out.vehicle_id[idx]) for idx in chosen)
    return PriorityVehicleAssignment(out, vehicle_ids, lane, gaps)


def select_yielding_vehicle_ids(
    state: TrafficState,
    *,
    fraction: float,
    seed: int,
    tag_as_av: bool = True,
) -> tuple[TrafficState, tuple[int, ...]]:
    """Select a reproducible non-PV subset for the fair B2-matched baseline."""

    if isinstance(fraction, bool) or not isinstance(
        fraction, (int, float, np.integer, np.floating)
    ):
        raise ValueError("fraction must be numeric in [0, 1]")
    fraction_f = float(fraction)
    if not 0.0 <= fraction_f <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an int")

    eligible = np.flatnonzero(state.alive & ~priority_vehicle_mask(state))
    count = int(np.floor(fraction_f * eligible.size))
    rng = np.random.default_rng(seed)
    # Always permute the full eligible population. Reusing the same seed for
    # several fractions then produces nested penetration sets.
    chosen = np.asarray(rng.permutation(eligible)[:count], dtype=np.int64)
    chosen.sort()
    out = state.copy()
    if tag_as_av and chosen.size:
        out.veh_type[chosen] = np.asarray(AV_VEH_TYPE, dtype=out.veh_type.dtype)
        out.behavior_id[chosen] = np.asarray(
            PRIORITY_YIELD_BEHAVIOR_ID, dtype=out.behavior_id.dtype
        )
        out.controlled[chosen] = False
    return out, tuple(int(out.vehicle_id[idx]) for idx in chosen)
