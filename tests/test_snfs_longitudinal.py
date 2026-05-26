import numpy as np
import pytest

from snfs_traffic.core import SimulationParams, TrafficState, build_occupancy, step_longitudinal_reference
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.topology import RingTopology


def _manual_state(*, lane, pos, vel=None, alive=None, controlled=None, length=None, changed_lane=None, last_lane_delta=None) -> TrafficState:
    n = len(lane)
    if vel is None:
        vel = [0] * n
    if alive is None:
        alive = [True] * n
    if controlled is None:
        controlled = [False] * n
    if length is None:
        length = [1] * n
    if changed_lane is None:
        changed_lane = [False] * n
    if last_lane_delta is None:
        last_lane_delta = [0] * n
    return TrafficState(
        vehicle_id=np.arange(n, dtype=np.int32),
        lane=np.array(lane, dtype=np.int16),
        pos=np.array(pos, dtype=np.int32),
        vel=np.array(vel, dtype=np.int16),
        length=np.array(length, dtype=np.int16),
        veh_type=np.zeros(n, dtype=np.int16),
        behavior_id=np.zeros(n, dtype=np.int16),
        alive=np.array(alive, dtype=np.bool_),
        last_lane_delta=np.array(last_lane_delta, dtype=np.int8),
        changed_lane=np.array(changed_lane, dtype=np.bool_),
        controlled=np.array(controlled, dtype=np.bool_),
    )


def _det_params(**kwargs) -> SimulationParams:
    base = dict(num_lanes=1, road_length=20, vmax_default=5, vmax_controlled=6, r=0.0, P2=1.0, P3=1.0, P4=0.0)
    base.update(kwargs)
    return SimulationParams(**base)


def test_public_import_callable() -> None:
    assert callable(step_longitudinal_reference)


def test_step_does_not_mutate_input_state() -> None:
    params = _det_params(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = _manual_state(lane=[0, 0], pos=[1, 5], vel=[1, 0])
    pos_before = state.pos.copy()
    vel_before = state.vel.copy()

    out = step_longitudinal_reference(state, params, topology, np.random.default_rng(123))

    assert out is not state
    np.testing.assert_array_equal(state.pos, pos_before)
    np.testing.assert_array_equal(state.vel, vel_before)
    assert not np.shares_memory(out.pos, state.pos)
    assert not np.shares_memory(out.vel, state.vel)


def test_single_vehicle_accelerates_to_vmax_default() -> None:
    params = _det_params()
    topology = RingTopology(num_lanes=1, length=20)
    state = _manual_state(lane=[0], pos=[0], vel=[0], controlled=[False])
    rng = np.random.default_rng(7)

    observed_vel = []
    observed_pos = []
    for _ in range(7):
        state = step_longitudinal_reference(state, params, topology, rng)
        observed_vel.append(int(state.vel[0]))
        observed_pos.append(int(state.pos[0]))

    assert observed_vel == [1, 2, 3, 4, 5, 5, 5]
    assert observed_pos == [1, 3, 6, 10, 15, 0, 5]


def test_controlled_vehicle_uses_vmax_controlled() -> None:
    params = _det_params()
    topology = RingTopology(num_lanes=1, length=20)
    state = _manual_state(lane=[0], pos=[0], vel=[0], controlled=[True])

    for _ in range(8):
        state = step_longitudinal_reference(state, params, topology, np.random.default_rng(9))

    assert int(state.vel[0]) == params.vmax_controlled


def test_close_front_vehicle_limits_speed_by_gap() -> None:
    params = _det_params(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = _manual_state(lane=[0, 0], pos=[2, 5], vel=[5, 0])

    out = step_longitudinal_reference(state, params, topology, np.random.default_rng(1))
    assert int(out.vel[0]) <= 2
    assert int(out.pos[0]) != int(out.pos[1])
    occupancy = build_occupancy(out, params)
    assert int((occupancy >= 0).sum()) == out.n_vehicles


def test_adjacent_front_vehicle_forces_zero_speed() -> None:
    params = _det_params(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = _manual_state(lane=[0, 0], pos=[2, 3], vel=[4, 0])

    out = step_longitudinal_reference(state, params, topology, np.random.default_rng(2))
    assert int(out.vel[0]) == 0
    assert int(out.pos[0]) == 2


def test_wraparound_front_vehicle_gap_handled() -> None:
    params = _det_params(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = _manual_state(lane=[0, 0], pos=[8, 0], vel=[5, 0])

    out = step_longitudinal_reference(state, params, topology, np.random.default_rng(3))
    assert int(out.vel[0]) <= 1
    assert int(out.pos[0]) in (8, 9)
    assert int(out.pos[0]) != int(out.pos[1])


def test_multilane_same_lane_only_interaction() -> None:
    params = _det_params(num_lanes=2, road_length=10)
    topology = RingTopology(num_lanes=2, length=10)
    state = _manual_state(lane=[0, 0, 1], pos=[2, 5, 3], vel=[5, 0, 0])

    out = step_longitudinal_reference(state, params, topology, np.random.default_rng(4))
    assert int(out.vel[0]) <= 2
    assert int(out.vel[2]) == 1


def test_inactive_vehicles_ignored_and_unchanged() -> None:
    params = _det_params(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = _manual_state(lane=[0, 0, 0], pos=[1, 2, 6], vel=[1, 4, 0], alive=[True, False, True])

    out = step_longitudinal_reference(state, params, topology, np.random.default_rng(5))
    assert int(out.pos[1]) == 2
    assert int(out.vel[1]) == 4
    # If inactive vehicle at pos=2 were incorrectly included, follower would be clamped to v=0.
    # Correct alive-only front neighbor is at pos=6 (gap=4), so follower does 1->2 and moves to 3.
    assert int(out.vel[0]) == 2
    assert int(out.pos[0]) == 3


def test_deterministic_same_seed_same_outputs() -> None:
    params = SimulationParams(num_lanes=4, road_length=60)
    topology = RingTopology(num_lanes=4, length=60)
    state = make_uniform_random_state(num_lanes=4, road_length=60, density=0.2, seed=123)

    a = step_longitudinal_reference(state.copy(), params, topology, np.random.default_rng(999))
    b = step_longitudinal_reference(state.copy(), params, topology, np.random.default_rng(999))

    for field in a.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_forced_random_braking_differs_from_disabled() -> None:
    topology = RingTopology(num_lanes=1, length=20)
    state = _manual_state(lane=[0], pos=[0], vel=[1])

    forced = step_longitudinal_reference(state, _det_params(P4=1.0), topology, np.random.default_rng(10))
    none = step_longitudinal_reference(state, _det_params(P4=0.0), topology, np.random.default_rng(10))

    assert int(forced.vel[0]) == 1  # 1->2 acceleration then -1 brake
    assert int(none.vel[0]) == 2


def test_q_and_p1_are_intentionally_ignored_by_step_longitudinal_reference() -> None:
    topology = RingTopology(num_lanes=1, length=20)
    state = _manual_state(lane=[0, 0], pos=[0, 5], vel=[1, 0])

    params_a = _det_params(q=0.99, P1=0.999, r=0.2, P2=0.9, P3=0.8, P4=0.1)
    params_b = _det_params(q=0.10, P1=0.10, r=0.2, P2=0.9, P3=0.8, P4=0.1)

    rng_a = np.random.default_rng(314)
    rng_b = np.random.default_rng(314)

    out_a = step_longitudinal_reference(state, params_a, topology, rng_a)
    out_b = step_longitudinal_reference(state, params_b, topology, rng_b)

    for field in out_a.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(out_a, field), getattr(out_b, field))
    assert rng_a.random() == rng_b.random()


def test_slow_to_start_can_be_forced() -> None:
    topology = RingTopology(num_lanes=1, length=20)
    state = _manual_state(lane=[0], pos=[0], vel=[0])

    blocked = step_longitudinal_reference(state, _det_params(r=1.0, P4=0.0), topology, np.random.default_rng(11))
    free = step_longitudinal_reference(state, _det_params(r=0.0, P4=0.0), topology, np.random.default_rng(11))

    assert int(blocked.vel[0]) == 0
    assert int(free.vel[0]) == 1


def test_velocity_bounds_over_multiple_steps() -> None:
    params = SimulationParams(num_lanes=4, road_length=100, vmax_default=5, vmax_controlled=6)
    topology = RingTopology(num_lanes=4, length=100)
    state = make_uniform_random_state(num_lanes=4, road_length=100, density=0.3, seed=123)
    rng = np.random.default_rng(321)

    for _ in range(20):
        state = step_longitudinal_reference(state, params, topology, rng)
        assert np.all(state.vel[state.alive] >= 0)
        assert np.all(state.vel[state.alive & ~state.controlled] <= params.vmax_default)
        assert np.all(state.vel[state.alive & state.controlled] <= params.vmax_controlled)
        assert np.all(state.pos[state.alive] >= 0)
        assert np.all(state.pos[state.alive] < params.road_length)


def test_no_duplicate_heads_across_densities() -> None:
    params = SimulationParams(num_lanes=2, road_length=30)
    topology = RingTopology(num_lanes=2, length=30)
    for density in (0.05, 0.2, 0.5, 1.0):
        state = make_uniform_random_state(num_lanes=2, road_length=30, density=density, seed=123)
        rng = np.random.default_rng(654)
        for _ in range(20):
            state = step_longitudinal_reference(state, params, topology, rng)
            occupancy = build_occupancy(state, params)
            assert int((occupancy >= 0).sum()) == int(state.alive.sum())


def test_density_one_full_occupancy_remains_valid() -> None:
    params = SimulationParams(num_lanes=2, road_length=5)
    topology = RingTopology(num_lanes=2, length=5)
    state = make_uniform_random_state(num_lanes=2, road_length=5, density=1.0, seed=123)

    out = step_longitudinal_reference(state, params, topology, np.random.default_rng(77))
    assert np.all(out.vel[out.alive] == 0)
    occupancy = build_occupancy(out, params)
    assert int((occupancy >= 0).sum()) == out.n_vehicles


def test_bus_length_ignored_intentionally_for_task6_head_cell_only() -> None:
    """Task 6 intentionally ignores length in occupancy/gaps; only head cells are occupied."""
    params = SimulationParams(num_lanes=1, road_length=20)
    topology = RingTopology(num_lanes=1, length=20)
    state = make_uniform_random_state(
        num_lanes=1,
        road_length=20,
        density=0.3,
        seed=123,
        vehicle_mix=VehicleMix(bus_fraction=1.0, bus_length=3),
    )

    out = step_longitudinal_reference(state, params, topology, np.random.default_rng(12))
    assert np.all(out.length == 3)
    occupancy = build_occupancy(out, params)
    assert int((occupancy >= 0).sum()) == out.n_vehicles


def test_lane_change_flags_are_reset() -> None:
    params = _det_params(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = _manual_state(lane=[0, 0], pos=[1, 4], changed_lane=[True, True], last_lane_delta=[-1, 1])

    out = step_longitudinal_reference(state, params, topology, np.random.default_rng(13))
    assert np.all(out.changed_lane == np.array([False, False]))
    assert np.all(out.last_lane_delta == np.array([0, 0], dtype=np.int8))


def test_invalid_rng_rejected() -> None:
    params = _det_params(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)
    state = _manual_state(lane=[0], pos=[0])

    with pytest.raises((TypeError, ValueError), match="rng|Generator"):
        step_longitudinal_reference(state, params, topology, rng=123)


def test_invalid_topology_rejected() -> None:
    params = _det_params(num_lanes=1, road_length=10)
    state = _manual_state(lane=[0], pos=[0])

    bad_topology = RingTopology(num_lanes=2, length=10)
    with pytest.raises(ValueError, match="num_lanes"):
        step_longitudinal_reference(state, params, bad_topology, np.random.default_rng(14))
