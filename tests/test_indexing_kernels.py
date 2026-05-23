import numpy as np
import pytest

from snfs_traffic.core import (
    INDEX_DTYPE,
    MISSING_INDEX,
    SimulationParams,
    build_lane_order,
    build_occupancy,
    compute_neighbors,
    empty_state,
)
from snfs_traffic.core.indexing_kernels import (
    build_lane_order_kernel,
    build_occupancy_kernel,
    compute_neighbors_kernel,
)
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.topology import RingTopology


def assert_indexing_outputs_equal(actual, expected):
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def test_build_occupancy_kernel_manual_equivalence():
    params = SimulationParams(num_lanes=2, road_length=10)
    state = empty_state(4)
    state.lane[:] = np.array([0, 0, 1, 1], dtype=state.lane.dtype)
    state.pos[:] = np.array([2, 5, 9, 1], dtype=state.pos.dtype)
    state.alive[:] = np.array([True, True, True, False], dtype=state.alive.dtype)

    expected = build_occupancy(state, params)
    actual = build_occupancy_kernel(state.lane, state.pos, state.alive, num_lanes=2, road_length=10)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype(INDEX_DTYPE)
    assert actual.shape == (2, 10)
    assert np.all(actual[:, 1] != 3)


def test_build_occupancy_duplicate_head_cell_rejected_in_wrapper_and_kernel():
    params = SimulationParams(num_lanes=2, road_length=10)
    state = empty_state(2)
    state.lane[:] = np.array([1, 1], dtype=state.lane.dtype)
    state.pos[:] = np.array([4, 4], dtype=state.pos.dtype)

    with pytest.raises(ValueError, match="duplicate|occupied"):
        build_occupancy(state, params)

    with pytest.raises(ValueError, match="duplicate|occupied"):
        build_occupancy_kernel(state.lane, state.pos, state.alive, num_lanes=2, road_length=10)


def test_build_lane_order_kernel_handcrafted_equivalence():
    occupancy = np.full((2, 10), MISSING_INDEX, dtype=INDEX_DTYPE)
    occupancy[0, 0] = 5
    occupancy[0, 3] = 2
    occupancy[0, 9] = 7
    occupancy[1, 4] = 1

    expected = build_lane_order(occupancy, n_vehicles=8)
    actual = build_lane_order_kernel(occupancy, n_vehicles=8)

    assert_indexing_outputs_equal(actual, expected)
    lane_order, lane_counts, lane_rank = actual
    np.testing.assert_array_equal(lane_counts, np.array([3, 1], dtype=INDEX_DTYPE))
    np.testing.assert_array_equal(lane_order[0, :3], np.array([5, 2, 7], dtype=INDEX_DTYPE))
    np.testing.assert_array_equal(lane_order[1, :1], np.array([1], dtype=INDEX_DTYPE))
    assert lane_rank[5] == 0
    assert lane_rank[2] == 1
    assert lane_rank[7] == 2
    assert lane_rank[1] == 0
    assert np.all(lane_rank[[0, 3, 4, 6]] == MISSING_INDEX)


def test_compute_neighbors_kernel_wraparound_equivalence():
    params = SimulationParams(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = empty_state(3)
    state.lane[:] = np.array([0, 0, 0], dtype=state.lane.dtype)
    state.pos[:] = np.array([0, 3, 9], dtype=state.pos.dtype)

    occupancy = build_occupancy(state, params)
    lane_order = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    expected = compute_neighbors(state, *lane_order, topology)
    actual = compute_neighbors_kernel(state.lane, state.pos, state.alive, *lane_order, road_length=10)

    assert_indexing_outputs_equal(actual, expected)
    front_id, back_id, front_gap, back_gap = actual
    assert front_id[2] == 0
    assert front_gap[2] == 0
    assert back_id[0] == 2
    assert back_gap[0] == 0


def test_indexing_kernels_random_state_equivalence_across_densities():
    num_lanes = 4
    road_length = 100
    for density in [0.0, 0.05, 0.2, 0.5, 0.8, 1.0]:
        for seed in [0, 1, 2, 3]:
            state = make_uniform_random_state(
                num_lanes=num_lanes,
                road_length=road_length,
                density=density,
                seed=seed,
            )
            params = SimulationParams(num_lanes=num_lanes, road_length=road_length)
            topology = RingTopology(num_lanes=num_lanes, length=road_length)

            occ_expected = build_occupancy(state, params)
            occ_actual = build_occupancy_kernel(
                state.lane,
                state.pos,
                state.alive,
                num_lanes=num_lanes,
                road_length=road_length,
            )
            np.testing.assert_array_equal(occ_actual, occ_expected)

            lo_expected = build_lane_order(occ_expected, n_vehicles=state.n_vehicles)
            lo_actual = build_lane_order_kernel(occ_actual, n_vehicles=state.n_vehicles)
            assert_indexing_outputs_equal(lo_actual, lo_expected)

            nb_expected = compute_neighbors(state, *lo_expected, topology)
            nb_actual = compute_neighbors_kernel(
                state.lane,
                state.pos,
                state.alive,
                *lo_actual,
                road_length=topology.length,
            )
            assert_indexing_outputs_equal(nb_actual, nb_expected)


def test_indexing_kernels_intentionally_ignore_bus_body_cells():
    mix = VehicleMix(bus_fraction=1.0, bus_length=3)
    state = make_uniform_random_state(num_lanes=3, road_length=30, density=0.4, seed=123, vehicle_mix=mix)
    params = SimulationParams(num_lanes=3, road_length=30)
    topology = RingTopology(num_lanes=3, length=30)

    assert np.all(state.length[state.alive] == 3)

    occ_expected = build_occupancy(state, params)
    occ_actual = build_occupancy_kernel(state.lane, state.pos, state.alive, num_lanes=3, road_length=30)
    np.testing.assert_array_equal(occ_actual, occ_expected)

    lo_expected = build_lane_order(occ_expected, n_vehicles=state.n_vehicles)
    lo_actual = build_lane_order_kernel(occ_actual, n_vehicles=state.n_vehicles)
    assert_indexing_outputs_equal(lo_actual, lo_expected)

    nb_expected = compute_neighbors(state, *lo_expected, topology)
    nb_actual = compute_neighbors_kernel(state.lane, state.pos, state.alive, *lo_actual, road_length=30)
    assert_indexing_outputs_equal(nb_actual, nb_expected)

    occupied_count = int((occ_actual >= 0).sum())
    alive_count = int(state.alive.sum())
    assert occupied_count == alive_count


def test_build_lane_order_wrapper_still_rejects_non_contiguous_occupancy():
    occupancy = np.full((2, 10), MISSING_INDEX, dtype=INDEX_DTYPE).T
    assert not occupancy.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        build_lane_order(occupancy, n_vehicles=4)
