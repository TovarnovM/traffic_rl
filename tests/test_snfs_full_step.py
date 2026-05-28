import numpy as np
import pytest

from snfs_traffic.core import (
    SimulationParams,
    TrafficState,
    build_occupancy,
    step_lane_change_reference,
    step_longitudinal_reference,
    step_reference,
    validate_state,
)
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
    d = dict(num_lanes=3, road_length=20, vmax_default=5, vmax_controlled=6, p_lane_change=1.0, r=0.0, P2=1.0, P3=1.0, P4=0.0)
    d.update(kwargs)
    return SimulationParams(**d)


def test_public_import():
    assert callable(step_reference)
    assert callable(step_longitudinal_reference)
    assert callable(step_lane_change_reference)


def test_full_step_does_not_mutate_input_state():
    s = _state(lane=[0, 0, 1], pos=[10, 11, 15], vel=[3, 0, 0])
    p = _params()
    t = RingTopology(num_lanes=3, length=20)
    lane0, pos0, vel0 = s.lane.copy(), s.pos.copy(), s.vel.copy()
    ch0, d0 = s.changed_lane.copy(), s.last_lane_delta.copy()

    out = step_reference(s, p, t, np.random.default_rng(2))

    assert out is not s
    np.testing.assert_array_equal(s.lane, lane0)
    np.testing.assert_array_equal(s.pos, pos0)
    np.testing.assert_array_equal(s.vel, vel0)
    np.testing.assert_array_equal(s.changed_lane, ch0)
    np.testing.assert_array_equal(s.last_lane_delta, d0)
    assert not np.shares_memory(out.lane, s.lane)
    assert not np.shares_memory(out.pos, s.pos)
    assert not np.shares_memory(out.vel, s.vel)
    assert not np.shares_memory(out.changed_lane, s.changed_lane)
    assert not np.shares_memory(out.last_lane_delta, s.last_lane_delta)


def test_full_step_matches_manual_composition_same_seed():
    state = _state(lane=[0, 0, 1], pos=[10, 11, 15], vel=[3, 0, 0])
    params = _params(p_lane_change=1.0)
    topology = RingTopology(3, 20)
    seed = 17

    rng_manual = np.random.default_rng(seed)
    after_lc = step_lane_change_reference(state, params, topology, rng_manual)
    saved_changed = after_lc.changed_lane.copy()
    saved_delta = after_lc.last_lane_delta.copy()
    after_long = step_longitudinal_reference(after_lc, params, topology, rng_manual)
    after_long.changed_lane = saved_changed
    after_long.last_lane_delta = saved_delta
    validate_state(after_long, params)

    rng_full = np.random.default_rng(seed)
    out = step_reference(state, params, topology, rng_full)

    for field in out.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(out, field), getattr(after_long, field))


def test_lane_change_happens_before_longitudinal_motion():
    params = _params(num_lanes=2, road_length=20, p_lane_change=1.0)
    topology = RingTopology(2, 20)
    state = _state(lane=[0, 0], pos=[10, 11], vel=[3, 0])

    out = step_reference(state, params, topology, np.random.default_rng(0))

    assert int(out.lane[0]) == 1
    assert int(out.vel[0]) == 4
    assert int(out.pos[0]) == 14
    assert bool(out.changed_lane[0]) is True
    assert int(out.last_lane_delta[0]) == 1


def test_lane_change_flags_survive_longitudinal_phase():
    state = _state(lane=[0, 0, 1], pos=[10, 11, 15], vel=[3, 0, 0])
    out = step_reference(state, _params(), RingTopology(3, 20), np.random.default_rng(4))

    assert bool(out.changed_lane[0]) is True
    assert int(out.last_lane_delta[0]) == 1
    assert bool(out.changed_lane[1]) is False
    assert int(out.last_lane_delta[1]) == 0
    assert bool(out.changed_lane[2]) is False
    assert int(out.last_lane_delta[2]) == 0


def test_one_lane_full_step_equals_longitudinal_step():
    state = _state(lane=[0, 0], pos=[1, 5], vel=[1, 0])
    params = _params(num_lanes=1)
    topology = RingTopology(1, 20)

    out_full = step_reference(state, params, topology, np.random.default_rng(123))
    out_long = step_longitudinal_reference(state, params, topology, np.random.default_rng(123))

    for field in out_full.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(out_full, field), getattr(out_long, field))


def test_p_lane_change_zero_blocks_lateral_but_moves_longitudinally():
    state = _state(lane=[0, 0, 1], pos=[10, 11, 15], vel=[3, 0, 0])
    out = step_reference(state, _params(p_lane_change=0.0), RingTopology(3, 20), np.random.default_rng(4))

    np.testing.assert_array_equal(out.lane, state.lane)
    assert not out.changed_lane.any()
    assert np.all(out.last_lane_delta == 0)
    assert not np.array_equal(out.pos, state.pos) or not np.array_equal(out.vel, state.vel)


def _conflict_state():
    return _state(lane=[0, 2, 0, 2], pos=[5, 5, 6, 6], vel=[2, 2, 0, 0])


def test_conflict_resolution_participates_in_full_step():
    params = _params()
    out = step_reference(_conflict_state(), params, RingTopology(3, 20), np.random.default_rng(7))

    moved = [int(out.lane[0] == 1 and out.changed_lane[0]), int(out.lane[1] == 1 and out.changed_lane[1])]
    assert sum(moved) == 1
    if moved[0] == 1:
        assert int(out.lane[1]) == 2
    else:
        assert int(out.lane[0]) == 0
    occ = build_occupancy(out, params)
    assert int((occ >= 0).sum()) == int(out.alive.sum())


def test_deterministic_same_seed_same_outputs():
    params = SimulationParams(num_lanes=4, road_length=60, p_lane_change=0.5)
    topology = RingTopology(num_lanes=4, length=60)
    state = make_uniform_random_state(num_lanes=4, road_length=60, density=0.2, seed=123)

    a = step_reference(state.copy(), params, topology, np.random.default_rng(999))
    b = step_reference(state.copy(), params, topology, np.random.default_rng(999))

    for field in a.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_different_seeds_can_produce_different_outcomes():
    state = _state(lane=[1, 1], pos=[10, 11], vel=[3, 0])
    params = _params()
    topology = RingTopology(3, 20)

    outcomes = set()
    for seed in range(51):
        out = step_reference(state, params, topology, np.random.default_rng(seed))
        outcomes.add((int(out.lane[0]), int(out.pos[0]), int(out.vel[0]), int(out.last_lane_delta[0])))
    assert len(outcomes) >= 2


def test_invalid_rng_rejected():
    s = _state(lane=[0], pos=[0])
    with pytest.raises((TypeError, ValueError), match="rng|Generator"):
        step_reference(s, _params(), RingTopology(3, 20), 123)  # type: ignore[arg-type]


def test_invalid_topology_rejected():
    s = _state(lane=[0], pos=[0])
    with pytest.raises(ValueError, match="topology"):
        step_reference(s, _params(road_length=20), RingTopology(3, 21), np.random.default_rng(0))


def test_full_random_rollout_remains_valid():
    for density in (0.05, 0.2, 0.5, 1.0):
        p = _params(num_lanes=4, road_length=100, p_lane_change=0.5)
        t = RingTopology(4, 100)
        s = make_uniform_random_state(num_lanes=4, road_length=100, density=density, seed=123)
        rng = np.random.default_rng(456)
        for _ in range(20):
            s = step_reference(s, p, t, rng)
            validate_state(s, p)
            occ = build_occupancy(s, p)
            assert int((occ >= 0).sum()) == int(s.alive.sum())
            assert np.all(s.lane[s.alive] >= 0)
            assert np.all(s.lane[s.alive] < p.num_lanes)
            assert np.all(s.pos[s.alive] >= 0)
            assert np.all(s.pos[s.alive] < p.road_length)
            assert np.all(s.vel[s.alive] >= 0)
            assert np.all(s.vel[s.alive & ~s.controlled] <= p.vmax_default)
            assert np.all(s.vel[s.alive & s.controlled] <= p.vmax_controlled)
            assert np.all(np.isin(s.last_lane_delta, [-1, 0, 1]))
            assert np.array_equal(s.changed_lane[s.alive], s.last_lane_delta[s.alive] != 0)


def test_density_one_full_occupancy_remains_valid():
    p = _params(num_lanes=2, road_length=5)
    s = make_uniform_random_state(num_lanes=2, road_length=5, density=1.0, seed=123)
    out = step_reference(s, p, RingTopology(2, 5), np.random.default_rng(12))
    assert np.all(out.vel[out.alive] == 0)
    occ = build_occupancy(out, p)
    assert int((occ >= 0).sum()) == int(out.alive.sum())


def test_bus_length_supported_with_body_aware_full_step():
    p = _params(num_lanes=3, road_length=30)
    s = make_uniform_random_state(
        num_lanes=3,
        road_length=30,
        density=0.3,
        seed=123,
        vehicle_mix=VehicleMix(bus_fraction=0.5, bus_length=3),
    )
    out = step_reference(s, p, RingTopology(3, 30), np.random.default_rng(2))
    assert np.any(out.length == 3)
    occ = build_occupancy(out, p)
    assert int((occ >= 0).sum()) == int(out.alive.sum())
