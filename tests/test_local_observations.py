import numpy as np

from snfs_traffic.core import SimulationParams, empty_state
from snfs_traffic.observations import LOCAL_OBSERVATION_FEATURES, LocalObservationConfig, build_local_observations
from snfs_traffic.simulator import TrafficSimulator
from snfs_traffic.topology import RingTopology


def test_no_controlled_shapes():
    p = SimulationParams(num_lanes=3, road_length=10)
    s = empty_state(2)
    s.pos[:] = [0, 1]
    s.controlled[:] = False
    o = build_local_observations(s, p, RingTopology(num_lanes=3, length=10))
    assert o.vehicle_id.shape == (0,)
    assert o.obs.shape == (0, 21)
    assert o.action_mask.shape == (0, 3)


def test_row_order_and_lane_boundaries():
    p = SimulationParams(num_lanes=3, road_length=10)
    s = empty_state(3)
    s.controlled[:] = [True, False, True]
    s.lane[:] = [0, 1, 2]
    s.pos[:] = [0, 3, 5]
    o = build_local_observations(s, p, RingTopology(num_lanes=3, length=10))
    np.testing.assert_array_equal(o.vehicle_id, np.array([0, 2], dtype=o.vehicle_id.dtype))
    assert bool(o.action_mask[0, 0]) is False
    assert bool(o.action_mask[1, 2]) is False


def test_occupied_adjacent_cell_no_crash_and_flags():
    p = SimulationParams(num_lanes=2, road_length=10)
    s = empty_state(2)
    s.controlled[:] = [True, False]
    s.lane[:] = [0, 1]
    s.pos[:] = [2, 2]
    s.vel[:] = [3, 1]
    obs = build_local_observations(s, p, RingTopology(num_lanes=2, length=10))
    i_cell = LOCAL_OBSERVATION_FEATURES.index("right_cell_free")
    i_safe = LOCAL_OBSERVATION_FEATURES.index("right_safe")
    assert obs.obs[0, i_cell] == 0.0
    assert obs.obs[0, i_safe] == 0.0
    assert bool(obs.action_mask[0, 2]) is False


def test_wraparound_and_relative_speed_and_include_position():
    p = SimulationParams(num_lanes=2, road_length=10)
    s = empty_state(2)
    s.controlled[:] = [True, False]
    s.lane[:] = [0, 0]
    s.pos[:] = [9, 1]
    s.vel[:] = [2, 5]
    obs = build_local_observations(s, p, RingTopology(num_lanes=2, length=10), LocalObservationConfig(include_position=False))
    i_fg = LOCAL_OBSERVATION_FEATURES.index("front_gap_norm")
    i_frs = LOCAL_OBSERVATION_FEATURES.index("front_rel_speed_norm")
    i_pos = LOCAL_OBSERVATION_FEATURES.index("ego_pos_norm")
    assert np.isclose(obs.obs[0, i_fg], 1 / 9)
    vmax = max(p.vmax_default, p.vmax_controlled, 1)
    assert np.isclose(obs.obs[0, i_frs], (5 - 2) / vmax)
    assert obs.obs[0, i_pos] == 0.0


def test_simulator_observe_matches_builder():
    p = SimulationParams(num_lanes=3, road_length=20)
    sim = TrafficSimulator(params=p)
    st = sim.reset()
    direct = build_local_observations(st, p, sim.topology)
    via = sim.observe()
    np.testing.assert_array_equal(direct.vehicle_id, via.vehicle_id)
    np.testing.assert_allclose(direct.obs, via.obs)
    np.testing.assert_array_equal(direct.action_mask, via.action_mask)
