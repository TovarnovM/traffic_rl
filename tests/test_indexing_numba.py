import numpy as np
import pytest

from snfs_traffic.core import (
    INDEX_DTYPE,
    MISSING_INDEX,
    SimulationParams,
    TrafficState,
    build_lane_order,
    build_occupancy,
    compute_neighbors,
)
from snfs_traffic.core.indexing_kernels import (
    build_lane_order_kernel,
    build_occupancy_kernel,
    compute_neighbors_kernel,
)
from snfs_traffic.core.indexing_numba import (
    NUMBA_AVAILABLE,
    build_lane_order_numba,
    build_occupancy_numba,
    compute_neighbors_numba,
)
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.topology import RingTopology

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba optional dependency is not installed")


def assert_tuple_arrays_equal(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def _make_state(lane, pos, alive):
    n = len(lane)
    return TrafficState(
        vehicle_id=np.arange(n, dtype=np.int32),
        lane=np.array(lane, dtype=np.int16),
        pos=np.array(pos, dtype=np.int32),
        vel=np.zeros(n, dtype=np.int16),
        length=np.ones(n, dtype=np.int16),
        veh_type=np.zeros(n, dtype=np.int16),
        behavior_id=np.zeros(n, dtype=np.int16),
        alive=np.array(alive, dtype=np.bool_),
        last_lane_delta=np.zeros(n, dtype=np.int8),
        changed_lane=np.zeros(n, dtype=np.bool_),
        controlled=np.zeros(n, dtype=np.bool_),
    )


def test_numba_build_occupancy_manual_equivalence():
    params = SimulationParams(num_lanes=2, road_length=10)
    state = _make_state([0, 0, 1, 1], [2, 5, 9, 1], [True, True, True, False])

    public_expected = build_occupancy(state, params)
    kernel_expected = build_occupancy_kernel(
        state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length
    )
    actual = build_occupancy_numba(
        state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length
    )

    np.testing.assert_array_equal(actual, public_expected)
    np.testing.assert_array_equal(actual, kernel_expected)
    assert actual.dtype == INDEX_DTYPE
    assert actual.shape == (2, 10)
    assert actual.flags.c_contiguous
    assert actual[1, 1] == MISSING_INDEX


def test_numba_build_occupancy_rejects_duplicate_head_cell():
    params = SimulationParams(num_lanes=2, road_length=10)
    state = _make_state([0, 0], [3, 3], [True, True])

    with pytest.raises(ValueError, match="duplicate|occupied"):
        build_occupancy(state, params)
    with pytest.raises(ValueError, match="duplicate|occupied"):
        build_occupancy_kernel(state.lane, state.pos, state.alive, num_lanes=2, road_length=10)
    with pytest.raises(ValueError):
        build_occupancy_numba(state.lane, state.pos, state.alive, num_lanes=2, road_length=10)


def test_numba_build_lane_order_handcrafted_occupancy_equivalence():
    occupancy = np.full((2, 10), MISSING_INDEX, dtype=INDEX_DTYPE)
    occupancy[0, 0] = 5
    occupancy[0, 3] = 2
    occupancy[0, 9] = 7
    occupancy[1, 4] = 1

    public_expected = build_lane_order(occupancy, n_vehicles=8)
    kernel_expected = build_lane_order_kernel(occupancy, n_vehicles=8)
    actual = build_lane_order_numba(occupancy, n_vehicles=8)

    assert_tuple_arrays_equal(actual, public_expected)
    assert_tuple_arrays_equal(actual, kernel_expected)

    lane_order, lane_counts, lane_rank = actual
    np.testing.assert_array_equal(lane_counts, np.array([3, 1], dtype=INDEX_DTYPE))
    np.testing.assert_array_equal(lane_order[0, :3], np.array([5, 2, 7], dtype=INDEX_DTYPE))
    np.testing.assert_array_equal(lane_order[1, :1], np.array([1], dtype=INDEX_DTYPE))
    assert lane_rank[5] == 0
    assert lane_rank[2] == 1
    assert lane_rank[7] == 2
    assert lane_rank[1] == 0
    assert lane_rank[0] == MISSING_INDEX
    assert lane_order.dtype == INDEX_DTYPE and lane_order.flags.c_contiguous
    assert lane_counts.dtype == INDEX_DTYPE and lane_counts.flags.c_contiguous
    assert lane_rank.dtype == INDEX_DTYPE and lane_rank.flags.c_contiguous


def test_numba_compute_neighbors_wraparound_equivalence():
    params = SimulationParams(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = _make_state([0, 0, 0], [0, 3, 9], [True, True, True])

    occupancy = build_occupancy(state, params)
    lane_order_public = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    lane_order_kernel = build_lane_order_kernel(occupancy, n_vehicles=state.n_vehicles)

    public_expected = compute_neighbors(state, *lane_order_public, topology)
    kernel_expected = compute_neighbors_kernel(state.lane, state.pos, state.alive, *lane_order_kernel, road_length=10)
    actual = compute_neighbors_numba(state.lane, state.pos, state.alive, *lane_order_kernel, road_length=10)

    assert_tuple_arrays_equal(actual, public_expected)
    assert_tuple_arrays_equal(actual, kernel_expected)

    front_id, back_id, front_gap, back_gap = actual
    assert front_id[2] == 0
    assert front_gap[2] == 0
    assert back_id[0] == 2
    assert back_gap[0] == 0


def test_numba_indexing_random_state_equivalence_across_densities():
    params = SimulationParams(num_lanes=4, road_length=100)
    topology = RingTopology(num_lanes=4, length=100)

    for density in [0.0, 0.05, 0.2, 0.5, 0.8, 1.0]:
        for seed in [0, 1, 2, 3]:
            state = make_uniform_random_state(num_lanes=4, road_length=100, density=density, seed=seed)
            occ_public = build_occupancy(state, params)
            occ_kernel = build_occupancy_kernel(
                state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length
            )
            occ_numba = build_occupancy_numba(
                state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length
            )
            np.testing.assert_array_equal(occ_numba, occ_public)
            np.testing.assert_array_equal(occ_numba, occ_kernel)

            lo_public = build_lane_order(occ_public, n_vehicles=state.n_vehicles)
            lo_kernel = build_lane_order_kernel(occ_kernel, n_vehicles=state.n_vehicles)
            lo_numba = build_lane_order_numba(occ_numba, n_vehicles=state.n_vehicles)
            assert_tuple_arrays_equal(lo_numba, lo_public)
            assert_tuple_arrays_equal(lo_numba, lo_kernel)

            nb_public = compute_neighbors(state, *lo_public, topology)
            nb_kernel = compute_neighbors_kernel(state.lane, state.pos, state.alive, *lo_kernel, road_length=topology.length)
            nb_numba = compute_neighbors_numba(state.lane, state.pos, state.alive, *lo_numba, road_length=topology.length)
            assert_tuple_arrays_equal(nb_numba, nb_public)
            assert_tuple_arrays_equal(nb_numba, nb_kernel)


def test_numba_indexing_intentionally_ignores_bus_body_cells():
    params = SimulationParams(num_lanes=3, road_length=30)
    topology = RingTopology(num_lanes=3, length=30)
    mix = VehicleMix(bus_fraction=1.0, bus_length=3)
    state = make_uniform_random_state(
        num_lanes=3,
        road_length=30,
        density=0.4,
        seed=123,
        vehicle_mix=mix,
    )

    assert np.all(state.length[state.alive] == 3)

    occ_public = build_occupancy(state, params)
    occ_numba = build_occupancy_numba(state.lane, state.pos, state.alive, num_lanes=3, road_length=30)
    np.testing.assert_array_equal(occ_numba, occ_public)
    assert int((occ_numba >= 0).sum()) == int(state.alive.sum())

    lo_public = build_lane_order(occ_public, n_vehicles=state.n_vehicles)
    lo_numba = build_lane_order_numba(occ_numba, n_vehicles=state.n_vehicles)
    assert_tuple_arrays_equal(lo_numba, lo_public)

    nb_public = compute_neighbors(state, *lo_public, topology)
    nb_numba = compute_neighbors_numba(state.lane, state.pos, state.alive, *lo_numba, road_length=topology.length)
    assert_tuple_arrays_equal(nb_numba, nb_public)


def test_numba_dispatchers_have_nopython_signatures():
    from snfs_traffic.core import indexing_numba as mod

    state = _make_state([0, 0], [1, 4], [True, True])
    occ = mod.build_occupancy_numba(state.lane, state.pos, state.alive, num_lanes=1, road_length=10)
    lo = mod.build_lane_order_numba(occ, n_vehicles=2)
    mod.compute_neighbors_numba(state.lane, state.pos, state.alive, *lo, road_length=10)

    assert mod._build_occupancy_numba_impl.nopython_signatures
    assert mod._build_lane_order_numba_impl.nopython_signatures
    assert mod._compute_neighbors_numba_impl.nopython_signatures


def test_public_core_import_works_independently_of_numba_specific_api():
    from snfs_traffic.core import build_occupancy as imported_build_occupancy

    assert callable(imported_build_occupancy)
