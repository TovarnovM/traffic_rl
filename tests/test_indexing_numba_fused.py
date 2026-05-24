import numpy as np
import pytest

from snfs_traffic.core import MISSING_INDEX, SimulationParams, TrafficState, build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.indexing_numba import NUMBA_AVAILABLE, build_index_and_neighbors_numba
from snfs_traffic.scenarios import make_uniform_random_state
from snfs_traffic.topology import RingTopology

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba optional dependency is not installed")


def _assert_fused_matches_reference(state: TrafficState, params: SimulationParams) -> None:
    topology = RingTopology(num_lanes=params.num_lanes, length=params.road_length)
    occ = build_occupancy(state, params)
    lo = build_lane_order(occ, n_vehicles=state.n_vehicles)
    nb = compute_neighbors(state, *lo, topology)
    fused = build_index_and_neighbors_numba(
        state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length
    )
    np.testing.assert_array_equal(fused[0], occ)
    np.testing.assert_array_equal(fused[1], lo[0])
    np.testing.assert_array_equal(fused[2], lo[1])
    np.testing.assert_array_equal(fused[3], lo[2])
    for i in range(4):
        np.testing.assert_array_equal(fused[4 + i], nb[i])


def test_fused_index_neighbors_random_density_single_and_multi_lane():
    for num_lanes in (1, 3):
        for density in (0.0, 0.1, 0.5, 1.0):
            state = make_uniform_random_state(num_lanes=num_lanes, road_length=40, density=density, seed=17)
            _assert_fused_matches_reference(state, SimulationParams(num_lanes=num_lanes, road_length=40))


def test_fused_index_neighbors_wraparound_manual_case():
    state = TrafficState(
        vehicle_id=np.array([0, 1, 2, 3], dtype=np.int32),
        lane=np.array([0, 0, 1, 1], dtype=np.int16),
        pos=np.array([0, 39, 1, 38], dtype=np.int32),
        vel=np.zeros(4, dtype=np.int16),
        length=np.ones(4, dtype=np.int16),
        veh_type=np.zeros(4, dtype=np.int16),
        behavior_id=np.zeros(4, dtype=np.int16),
        alive=np.array([True, True, True, True], dtype=np.bool_),
        last_lane_delta=np.zeros(4, dtype=np.int8),
        changed_lane=np.zeros(4, dtype=np.bool_),
        controlled=np.zeros(4, dtype=np.bool_),
    )
    _assert_fused_matches_reference(state, SimulationParams(num_lanes=2, road_length=40))


def test_fused_index_neighbors_inactive_vehicle_ignored_and_single_vehicle_lane():
    state = TrafficState(
        vehicle_id=np.array([0, 1, 2], dtype=np.int32),
        lane=np.array([0, 0, 1], dtype=np.int16),
        pos=np.array([5, 5, 10], dtype=np.int32),
        vel=np.zeros(3, dtype=np.int16),
        length=np.ones(3, dtype=np.int16),
        veh_type=np.zeros(3, dtype=np.int16),
        behavior_id=np.zeros(3, dtype=np.int16),
        alive=np.array([True, False, True], dtype=np.bool_),
        last_lane_delta=np.zeros(3, dtype=np.int8),
        changed_lane=np.zeros(3, dtype=np.bool_),
        controlled=np.zeros(3, dtype=np.bool_),
    )
    params = SimulationParams(num_lanes=2, road_length=20)
    fused = build_index_and_neighbors_numba(state.lane, state.pos, state.alive, num_lanes=2, road_length=20)
    assert fused[0][0, 5] == 0
    assert fused[0][1, 10] == 2
    assert fused[0][0, 6] == MISSING_INDEX
    _assert_fused_matches_reference(state, params)


def test_fused_duplicate_head_cell_raises_like_reference():
    state = TrafficState(
        vehicle_id=np.array([0, 1], dtype=np.int32),
        lane=np.array([0, 0], dtype=np.int16),
        pos=np.array([3, 3], dtype=np.int32),
        vel=np.zeros(2, dtype=np.int16),
        length=np.ones(2, dtype=np.int16),
        veh_type=np.zeros(2, dtype=np.int16),
        behavior_id=np.zeros(2, dtype=np.int16),
        alive=np.array([True, True], dtype=np.bool_),
        last_lane_delta=np.zeros(2, dtype=np.int8),
        changed_lane=np.zeros(2, dtype=np.bool_),
        controlled=np.zeros(2, dtype=np.bool_),
    )
    params = SimulationParams(num_lanes=1, road_length=10)
    with pytest.raises(ValueError, match="duplicate|occupied"):
        build_occupancy(state, params)
    with pytest.raises(ValueError, match="duplicate|occupied"):
        build_index_and_neighbors_numba(state.lane, state.pos, state.alive, num_lanes=1, road_length=10)
