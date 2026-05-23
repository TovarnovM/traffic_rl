import numpy as np
import pytest

from snfs_traffic.core import SimulationParams, TrafficState, build_occupancy, step_lane_change_reference, validate_state
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.topology import RingTopology


def _state(lane, pos, vel=None, alive=None, controlled=None, length=None, changed_lane=None, last_lane_delta=None):
    n = len(lane)
    return TrafficState(
        vehicle_id=np.arange(n, dtype=np.int32),
        lane=np.asarray(lane, dtype=np.int16),
        pos=np.asarray(pos, dtype=np.int32),
        vel=np.asarray([0] * n if vel is None else vel, dtype=np.int16),
        length=np.asarray([1] * n if length is None else length, dtype=np.int16),
        veh_type=np.zeros(n, dtype=np.int16),
        behavior_id=np.zeros(n, dtype=np.int16),
        alive=np.asarray([True] * n if alive is None else alive, dtype=np.bool_),
        last_lane_delta=np.asarray([0] * n if last_lane_delta is None else last_lane_delta, dtype=np.int8),
        changed_lane=np.asarray([False] * n if changed_lane is None else changed_lane, dtype=np.bool_),
        controlled=np.asarray([False] * n if controlled is None else controlled, dtype=np.bool_),
    )


def _params(**kwargs):
    d = dict(num_lanes=3, road_length=20, vmax_default=5, vmax_controlled=6, p_lane_change=1.0)
    d.update(kwargs)
    return SimulationParams(**d)


def test_public_import():
    assert callable(step_lane_change_reference)


def test_does_not_mutate_input_state():
    s = _state(lane=[0, 0, 1], pos=[10, 11, 15], vel=[3, 0, 0])
    p = _params()
    t = RingTopology(num_lanes=3, length=20)
    lane0, pos0, vel0 = s.lane.copy(), s.pos.copy(), s.vel.copy()
    ch0, d0 = s.changed_lane.copy(), s.last_lane_delta.copy()
    out = step_lane_change_reference(s, p, t, np.random.default_rng(2))
    assert out is not s
    np.testing.assert_array_equal(s.lane, lane0)
    np.testing.assert_array_equal(s.pos, pos0)
    np.testing.assert_array_equal(s.vel, vel0)
    np.testing.assert_array_equal(s.changed_lane, ch0)
    np.testing.assert_array_equal(s.last_lane_delta, d0)
    assert not np.shares_memory(out.lane, s.lane)


def test_p_lane_change_default():
    assert SimulationParams(num_lanes=1, road_length=1).p_lane_change == 0.5


def test_no_changes_one_lane():
    s = _state(lane=[0, 0], pos=[1, 5], vel=[3, 0])
    out = step_lane_change_reference(s, _params(num_lanes=1), RingTopology(num_lanes=1, length=20), np.random.default_rng(1))
    np.testing.assert_array_equal(out.lane, s.lane)
    assert not out.changed_lane.any()
    assert np.all(out.last_lane_delta == 0)


def test_no_lateral_wrapping_edges():
    s = _state(lane=[0, 2], pos=[10, 12], vel=[3, 3])
    out = step_lane_change_reference(s, _params(num_lanes=3), RingTopology(num_lanes=3, length=20), np.random.default_rng(0))
    assert np.all(out.lane >= 0)
    assert np.all(out.lane < 3)
    validate_state(out, _params(num_lanes=3))


def test_forced_change_when_eligible():
    s = _state(lane=[0, 0, 1], pos=[10, 11, 15], vel=[3, 0, 0])
    out = step_lane_change_reference(s, _params(), RingTopology(3, 20), np.random.default_rng(4))
    assert int(out.lane[0]) == 1
    assert int(out.pos[0]) == 10 and int(out.vel[0]) == 3
    assert out.changed_lane[0] and int(out.last_lane_delta[0]) == 1
    occ = build_occupancy(out, _params())
    assert int((occ >= 0).sum()) == int(out.alive.sum())


def test_p_zero_disables_attempt():
    s = _state(lane=[0, 0, 1], pos=[10, 11, 15], vel=[3, 0, 0])
    out = step_lane_change_reference(s, _params(p_lane_change=0.0), RingTopology(3, 20), np.random.default_rng(4))
    np.testing.assert_array_equal(out.lane, s.lane)
    assert not out.changed_lane.any()


def test_incentive_blocks_change():
    s = _state(lane=[0, 0, 1], pos=[10, 11, 12], vel=[3, 0, 0])
    out = step_lane_change_reference(s, _params(), RingTopology(3, 20), np.random.default_rng(4))
    assert int(out.lane[0]) == 0


def test_safety_blocks_change():
    s = _state(lane=[0, 0, 1, 1], pos=[10, 11, 15, 9], vel=[3, 0, 5, 5])
    out = step_lane_change_reference(s, _params(), RingTopology(3, 20), np.random.default_rng(4))
    assert int(out.lane[0]) == 0


def test_target_head_occupied_blocks_change():
    s = _state(lane=[0, 0, 1], pos=[10, 11, 10], vel=[3, 0, 0])
    out = step_lane_change_reference(s, _params(), RingTopology(3, 20), np.random.default_rng(0))
    assert int(out.lane[0]) == 0


def test_empty_target_lane_handled():
    s = _state(lane=[0, 0], pos=[10, 11], vel=[3, 0])
    out = step_lane_change_reference(s, _params(num_lanes=2), RingTopology(2, 20), np.random.default_rng(0))
    assert int(out.lane[0]) == 1


def test_periodic_target_gaps_wrap_logic():
    s = _state(lane=[0, 0, 1, 1], pos=[8, 9, 3, 5], vel=[3, 0, 0, 1])
    out = step_lane_change_reference(s, _params(num_lanes=2, road_length=10), RingTopology(2, 10), np.random.default_rng(0))
    assert int(out.lane[0]) == 1


def test_present_lane_missing_front_free_road_not_incentivized():
    s = _state(lane=[0], pos=[3], vel=[2])
    out = step_lane_change_reference(s, _params(num_lanes=2), RingTopology(2, 20), np.random.default_rng(0))
    assert int(out.lane[0]) == 0


def test_both_adjacent_eligible_stochastic_side_tiebreak():
    s = _state(lane=[1, 1], pos=[10, 11], vel=[3, 0])
    seen = set()
    for seed in range(51):
        out = step_lane_change_reference(s, _params(), RingTopology(3, 20), np.random.default_rng(seed))
        seen.add(int(out.lane[0]))
    assert seen == {0, 2}


def _conflict_state():
    return _state(lane=[0, 2, 0, 2], pos=[5, 5, 6, 6], vel=[2, 2, 0, 0])


def test_same_target_conflict_one_winner():
    s = _conflict_state()
    out = step_lane_change_reference(s, _params(), RingTopology(3, 20), np.random.default_rng(7))
    moved = [int(out.lane[0] == 1), int(out.lane[1] == 1)]
    assert sum(moved) == 1
    occ = build_occupancy(out, _params())
    assert int((occ >= 0).sum()) == int(out.alive.sum())


def test_conflict_deterministic_same_seed():
    s = _conflict_state()
    a = step_lane_change_reference(s.copy(), _params(), RingTopology(3, 20), np.random.default_rng(9))
    b = step_lane_change_reference(s.copy(), _params(), RingTopology(3, 20), np.random.default_rng(9))
    for f in a.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))


def test_conflict_winner_differs_across_seeds():
    winners = set()
    for seed in range(51):
        out = step_lane_change_reference(_conflict_state(), _params(), RingTopology(3, 20), np.random.default_rng(seed))
        winners.add(0 if out.lane[0] == 1 else 1)
    assert winners == {0, 1}


def test_inactive_vehicles_ignored():
    s = _state(lane=[0, 0, 1], pos=[10, 11, 10], vel=[3, 0, 0], alive=[True, True, False])
    out = step_lane_change_reference(s, _params(), RingTopology(3, 20), np.random.default_rng(0))
    assert int(out.lane[0]) == 1
    assert int(out.lane[2]) == 1 and int(out.pos[2]) == 10 and int(out.vel[2]) == 0


def test_flags_reset_for_non_movers():
    s = _state(lane=[0, 0], pos=[1, 2], vel=[0, 0], changed_lane=[True, True], last_lane_delta=[-1, 1], alive=[True, False])
    out = step_lane_change_reference(s, _params(num_lanes=1), RingTopology(1, 20), np.random.default_rng(0))
    assert np.all(out.changed_lane == np.array([False, False]))
    assert np.all(out.last_lane_delta == np.array([0, 0], dtype=np.int8))


def test_multilane_random_sim_valid():
    for density in (0.05, 0.2, 0.5, 1.0):
        p = _params(num_lanes=4, road_length=100, p_lane_change=0.5)
        t = RingTopology(4, 100)
        s = make_uniform_random_state(num_lanes=4, road_length=100, density=density, seed=123)
        rng = np.random.default_rng(456)
        for _ in range(20):
            p0, v0 = s.pos.copy(), s.vel.copy()
            s = step_lane_change_reference(s, p, t, rng)
            validate_state(s, p)
            occ = build_occupancy(s, p)
            assert int((occ >= 0).sum()) == int(s.alive.sum())
            assert np.all(s.lane[s.alive] >= 0)
            assert np.all(s.lane[s.alive] < p.num_lanes)
            np.testing.assert_array_equal(s.pos, p0)
            np.testing.assert_array_equal(s.vel, v0)


def test_density_one_full_occupancy_no_changes():
    p = _params(num_lanes=2, road_length=5)
    s = make_uniform_random_state(num_lanes=2, road_length=5, density=1.0, seed=123)
    out = step_lane_change_reference(s, p, RingTopology(2, 5), np.random.default_rng(12))
    np.testing.assert_array_equal(out.lane, s.lane)
    np.testing.assert_array_equal(out.pos, s.pos)
    np.testing.assert_array_equal(out.vel, s.vel)


def test_bus_length_ignored_intentionally_for_task7_head_cell_only():
    """Length is intentionally ignored; lane-change uses only head-cell occupancy/gaps."""
    p = _params(num_lanes=3, road_length=20)
    s = make_uniform_random_state(
        num_lanes=3,
        road_length=20,
        density=0.3,
        seed=123,
        vehicle_mix=VehicleMix(bus_fraction=1.0, bus_length=3),
    )
    out = step_lane_change_reference(s, p, RingTopology(3, 20), np.random.default_rng(2))
    assert np.all(out.length == 3)
    occ = build_occupancy(out, p)
    assert int((occ >= 0).sum()) == int(out.alive.sum())


def test_invalid_topology_rejected():
    s = _state(lane=[0], pos=[0])
    with pytest.raises(ValueError, match="topology"):
        step_lane_change_reference(s, _params(road_length=20), RingTopology(3, 21), np.random.default_rng(0))


class _BadTopology:
    boundary = "open"
    num_lanes = 3
    length = 20


def test_invalid_topology_boundary_rejected():
    s = _state(lane=[0], pos=[0])
    with pytest.raises(ValueError):
        step_lane_change_reference(s, _params(), _BadTopology(), np.random.default_rng(0))  # type: ignore[arg-type]


def test_invalid_rng_rejected():
    s = _state(lane=[0], pos=[0])
    with pytest.raises((TypeError, ValueError), match="rng|Generator"):
        step_lane_change_reference(s, _params(), RingTopology(3, 20), 123)  # type: ignore[arg-type]
