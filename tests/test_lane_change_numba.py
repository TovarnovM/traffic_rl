from __future__ import annotations

import numpy as np
import pytest

from snfs_traffic.core import SimulationParams, step_lane_change_reference
from snfs_traffic.core.indexing import build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.lane_change_numba import NUMBA_AVAILABLE, collect_lane_change_proposals_numba
from snfs_traffic.scenarios import make_uniform_random_state
from snfs_traffic.topology import RingTopology


def test_lane_change_numba_optional_import() -> None:
    assert isinstance(NUMBA_AVAILABLE, bool)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba unavailable")
@pytest.mark.parametrize("p_lane_change", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("num_lanes,road_length,density", [(1, 40, 0.0), (2, 80, 0.05), (3, 80, 0.5), (4, 40, 1.0)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lane_change_numba_matches_reference_final_state(
    p_lane_change: float,
    num_lanes: int,
    road_length: int,
    density: float,
    seed: int,
) -> None:
    params = SimulationParams(num_lanes=num_lanes, road_length=road_length, p_lane_change=p_lane_change)
    topology = RingTopology(num_lanes=num_lanes, length=road_length)
    state = make_uniform_random_state(num_lanes=num_lanes, road_length=road_length, density=density, seed=seed)

    rng_ref = np.random.default_rng(seed + 100)
    rng_numba = np.random.default_rng(seed + 100)

    ref = step_lane_change_reference(state.copy(), params, topology, rng_ref)

    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, _, front_gap, _ = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

    proposals = collect_lane_change_proposals_numba(
        lane=state.lane,
        pos=state.pos,
        vel=state.vel,
        alive=state.alive,
        controlled=state.controlled,
        occupancy=occupancy,
        lane_order=lane_order,
        lane_counts=lane_counts,
        front_id=front_id,
        front_gap=front_gap,
        num_lanes=params.num_lanes,
        road_length=params.road_length,
        vmax_default=params.vmax_default,
        vmax_controlled=params.vmax_controlled,
        p_lane_change=params.p_lane_change,
        rng=rng_numba,
    )

    # reproduce reference accept/apply using same RNG stream
    accepted = {}
    for (target_lane, _), candidates in proposals.items():
        if len(candidates) == 1:
            winner = candidates[0]
        else:
            winner = candidates[int(rng_numba.integers(0, len(candidates)))]
        accepted[winner] = target_lane

    out = state.copy()
    out.changed_lane.fill(False)
    out.last_lane_delta.fill(0)
    for i, target_lane in accepted.items():
        old_lane = int(out.lane[i])
        out.lane[i] = target_lane
        out.changed_lane[i] = True
        out.last_lane_delta[i] = target_lane - old_lane

    for field in ("lane", "changed_lane", "last_lane_delta", "alive", "controlled"):
        np.testing.assert_array_equal(getattr(out, field), getattr(ref, field))

    assert float(rng_ref.random()) == float(rng_numba.random())


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba unavailable")
def test_lane_change_numba_conflict_heavy_rng_parity() -> None:
    params = SimulationParams(num_lanes=2, road_length=6, p_lane_change=1.0)
    topology = RingTopology(num_lanes=2, length=6)
    # deterministic, dense, conflict-heavy candidate layout from random state with fixed seed
    state = make_uniform_random_state(num_lanes=2, road_length=6, density=0.5, seed=123)

    rng_ref = np.random.default_rng(777)
    rng_numba = np.random.default_rng(777)
    _ = step_lane_change_reference(state.copy(), params, topology, rng_ref)

    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, _, front_gap, _ = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)
    _ = collect_lane_change_proposals_numba(
        lane=state.lane,
        pos=state.pos,
        vel=state.vel,
        alive=state.alive,
        controlled=state.controlled,
        occupancy=occupancy,
        lane_order=lane_order,
        lane_counts=lane_counts,
        front_id=front_id,
        front_gap=front_gap,
        num_lanes=params.num_lanes,
        road_length=params.road_length,
        vmax_default=params.vmax_default,
        vmax_controlled=params.vmax_controlled,
        p_lane_change=params.p_lane_change,
        rng=rng_numba,
    )
    # parity only check here; full state check done in parametrized test.
    assert float(rng_ref.random()) == float(rng_numba.random())
