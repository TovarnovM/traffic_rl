import numpy as np
import pytest

from snfs_traffic.core import (
    INDEX_DTYPE,
    MISSING_GAP,
    MISSING_INDEX,
    SimulationParams,
    TrafficState,
    build_lane_order,
    build_occupancy,
    compute_neighbors,
)
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.topology import RingTopology


def _manual_state(*, lane, pos, alive=None, length=None) -> TrafficState:
    n = len(lane)
    if alive is None:
        alive = [True] * n
    if length is None:
        length = [1] * n
    return TrafficState(
        vehicle_id=np.arange(n, dtype=np.int32),
        lane=np.array(lane, dtype=np.int16),
        pos=np.array(pos, dtype=np.int32),
        vel=np.zeros(n, dtype=np.int16),
        length=np.array(length, dtype=np.int16),
        veh_type=np.zeros(n, dtype=np.int16),
        behavior_id=np.zeros(n, dtype=np.int16),
        alive=np.array(alive, dtype=np.bool_),
        last_lane_delta=np.zeros(n, dtype=np.int8),
        changed_lane=np.zeros(n, dtype=np.bool_),
        controlled=np.zeros(n, dtype=np.bool_),
    )


def test_build_occupancy_basic_manual_state() -> None:
    params = SimulationParams(num_lanes=2, road_length=10)
    state = _manual_state(lane=[0, 0, 1], pos=[2, 5, 9])

    occupancy = build_occupancy(state, params)

    assert occupancy.shape == (2, 10)
    assert occupancy.dtype == np.int32
    assert occupancy.flags.c_contiguous
    assert occupancy[0, 2] == 0
    assert occupancy[0, 5] == 1
    assert occupancy[1, 9] == 2
    assert int((occupancy >= 0).sum()) == 3


def test_build_occupancy_ignores_inactive() -> None:
    params = SimulationParams(num_lanes=1, road_length=6)
    state = _manual_state(lane=[0, 0], pos=[1, 2], alive=[True, False])

    occupancy = build_occupancy(state, params)

    assert occupancy[0, 1] == 0
    assert occupancy[0, 2] == -1
    assert not np.any(occupancy == 1)


def test_build_occupancy_duplicate_head_cells_rejected() -> None:
    params = SimulationParams(num_lanes=1, road_length=10)
    state = _manual_state(lane=[0, 0], pos=[3, 3])

    with pytest.raises(ValueError, match="duplicate|occupied"):
        build_occupancy(state, params)


def test_build_lane_order_basic_ordering() -> None:
    occupancy = np.full((2, 10), -1, dtype=np.int32)
    occupancy[0, 0] = 5
    occupancy[0, 3] = 2
    occupancy[0, 9] = 7
    occupancy[1, 4] = 1

    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=8)

    assert lane_order.shape == (2, 10)
    assert lane_order.dtype == np.int32
    assert lane_counts.dtype == np.int32
    assert lane_rank.dtype == np.int32
    assert lane_order.flags.c_contiguous
    assert lane_counts.flags.c_contiguous
    assert lane_rank.flags.c_contiguous

    np.testing.assert_array_equal(lane_counts, np.array([3, 1], dtype=np.int32))
    np.testing.assert_array_equal(lane_order[0, :3], np.array([5, 2, 7], dtype=np.int32))
    np.testing.assert_array_equal(lane_order[1, :1], np.array([1], dtype=np.int32))
    assert np.all(lane_order[0, 3:] == -1)
    assert np.all(lane_order[1, 1:] == -1)
    assert lane_rank[5] == 0
    assert lane_rank[2] == 1
    assert lane_rank[7] == 2
    assert lane_rank[1] == 0
    assert lane_rank[0] == -1
    assert lane_rank[3] == -1


def test_build_lane_order_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="occupancy"):
        build_lane_order([[0, 1]], n_vehicles=2)

    with pytest.raises(ValueError, match="occupancy"):
        build_lane_order(np.array([0, 1], dtype=np.int32), n_vehicles=2)

    with pytest.raises(ValueError, match="dtype"):
        build_lane_order(np.array([[0, -1]], dtype=np.int64), n_vehicles=2)

    bad = np.full((2, 4), -1, dtype=np.int32).T
    assert not bad.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        build_lane_order(bad, n_vehicles=3)

    bad = np.array([[-2]], dtype=np.int32)
    with pytest.raises(ValueError, match="< -1"):
        build_lane_order(bad, n_vehicles=3)

    bad = np.array([[3]], dtype=np.int32)
    with pytest.raises(ValueError, match=">= n_vehicles"):
        build_lane_order(bad, n_vehicles=3)

    bad = np.array([[1, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="duplicate"):
        build_lane_order(bad, n_vehicles=3)

    with pytest.raises(ValueError, match="n_vehicles"):
        build_lane_order(np.full((1, 1), -1, dtype=np.int32), n_vehicles=True)

    with pytest.raises(ValueError, match="n_vehicles"):
        build_lane_order(np.full((1, 1), -1, dtype=np.int32), n_vehicles=-1)


def test_compute_neighbors_single_vehicle_lane_policy() -> None:
    params = SimulationParams(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = _manual_state(lane=[0], pos=[4])
    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)

    front_id, back_id, front_gap, back_gap = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

    assert front_id[0] == -1
    assert back_id[0] == -1
    assert front_gap[0] == -1
    assert back_gap[0] == -1


def test_compute_neighbors_two_vehicles_same_lane() -> None:
    params = SimulationParams(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = _manual_state(lane=[0, 0], pos=[2, 5])
    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)

    front_id, back_id, front_gap, back_gap = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

    assert front_id[0] == 1
    assert back_id[0] == 1
    assert front_gap[0] == 2
    assert back_gap[0] == 6

    assert front_id[1] == 0
    assert back_id[1] == 0
    assert front_gap[1] == 6
    assert back_gap[1] == 2


def test_compute_neighbors_wraparound() -> None:
    params = SimulationParams(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = _manual_state(lane=[0, 0, 0], pos=[0, 3, 9])
    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, back_id, front_gap, back_gap = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

    assert front_id[2] == 0
    assert front_gap[2] == 0
    assert back_id[0] == 2
    assert back_gap[0] == 0
    assert front_id[0] == 1
    assert front_gap[0] == 2


def test_compute_neighbors_multiple_lanes_independent() -> None:
    params = SimulationParams(num_lanes=2, road_length=10)
    topology = RingTopology(num_lanes=2, length=10)
    state = _manual_state(lane=[0, 0, 1, 1], pos=[1, 5, 2, 9])
    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, back_id, _, _ = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

    assert set([front_id[0], back_id[0]]) == {1}
    assert set([front_id[1], back_id[1]]) == {0}
    assert set([front_id[2], back_id[2]]) == {3}
    assert set([front_id[3], back_id[3]]) == {2}


def test_compute_neighbors_ignores_inactive() -> None:
    params = SimulationParams(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = _manual_state(lane=[0, 0, 0], pos=[1, 3, 5], alive=[True, False, True])
    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, back_id, front_gap, back_gap = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

    assert front_id[1] == -1 and back_id[1] == -1
    assert front_gap[1] == -1 and back_gap[1] == -1
    assert set([front_id[0], back_id[0]]) == {2}
    assert set([front_id[2], back_id[2]]) == {0}


def test_generated_random_state_indexing_invariants() -> None:
    state = make_uniform_random_state(num_lanes=4, road_length=100, density=0.3, seed=123)
    params = SimulationParams(num_lanes=4, road_length=100)
    topology = RingTopology(num_lanes=4, length=100)

    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, back_id, front_gap, back_gap = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

    assert int((occupancy >= 0).sum()) == state.n_vehicles
    assert int(lane_counts.sum()) == state.n_vehicles
    assert np.all(lane_rank[state.alive] >= 0)

    for arr in (front_id, back_id, front_gap, back_gap):
        assert arr.shape == (state.n_vehicles,)
        assert arr.dtype == np.int32

    assert np.all((front_gap == -1) | (front_gap >= 0))
    assert np.all((back_gap == -1) | (back_gap >= 0))

    for i in range(state.n_vehicles):
        if not state.alive[i]:
            continue
        if front_id[i] != -1:
            assert back_id[front_id[i]] == i
        if back_id[i] != -1:
            assert front_id[back_id[i]] == i


def test_density_one_full_occupancy_indexing() -> None:
    state = make_uniform_random_state(num_lanes=2, road_length=5, density=1.0, seed=123)
    params = SimulationParams(num_lanes=2, road_length=5)
    topology = RingTopology(num_lanes=2, length=5)

    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, back_id, front_gap, back_gap = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

    assert not np.any(occupancy == -1)
    np.testing.assert_array_equal(lane_counts, np.array([5, 5], dtype=np.int32))
    assert np.all(front_gap[state.alive] == 0)
    assert np.all(back_gap[state.alive] == 0)
    assert np.all(front_id[state.alive] >= 0)
    assert np.all(back_id[state.alive] >= 0)
    assert np.all(front_id[state.alive] != np.arange(state.n_vehicles, dtype=np.int32))


def test_bus_length_unplaceable_configuration_rejected() -> None:
    mix = VehicleMix(bus_fraction=1.0, bus_length=3)
    import pytest
    with pytest.raises(ValueError, match="non-overlapping"):
        make_uniform_random_state(num_lanes=1, road_length=10, density=0.5, seed=123, vehicle_mix=mix)
    params = SimulationParams(num_lanes=1, road_length=10)

    occupancy = build_occupancy(state, params)

    assert np.all(state.length == 3)
    assert int((occupancy >= 0).sum()) == state.n_vehicles


def test_public_imports_indexing_api() -> None:
    assert INDEX_DTYPE == np.int32
    assert MISSING_INDEX == -1
    assert MISSING_GAP == -1
    assert callable(build_occupancy)
    assert callable(build_lane_order)
    assert callable(compute_neighbors)
