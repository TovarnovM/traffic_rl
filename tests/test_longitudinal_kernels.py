import numpy as np

from snfs_traffic.core import SimulationParams, build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.longitudinal_kernels import advance_positions_kernel, compute_longitudinal_velocities_kernel
from snfs_traffic.scenarios import make_uniform_random_state
from snfs_traffic.topology import RingTopology


def _legacy_compute_longitudinal_arrays(
    vel,
    pos,
    alive,
    controlled,
    front_id,
    front_gap,
    *,
    road_length,
    vmax_default,
    vmax_controlled,
    G,
    S,
    r,
    P2,
    P3,
    P4,
    rng,
):
    out_vel = vel.astype(np.int64, copy=True)
    for i in range(vel.shape[0]):
        if not alive[i]:
            continue
        old_v = int(vel[i])
        vmax_i = int(vmax_controlled if controlled[i] else vmax_default)

        if int(front_id[i]) == -1:
            gap_i = int(road_length - 1)
            leader_v = vmax_i
        else:
            gap_i = int(front_gap[i])
            leader_v = int(vel[int(front_id[i])])

        v = min(old_v + 1, vmax_i)
        if int(front_id[i]) != -1 and gap_i <= G:
            anticipated_leader_motion = max(leader_v - int(S), 0)
            anticipated_gap = gap_i + anticipated_leader_motion
            v = min(v, anticipated_gap)

        if old_v == 0 and v > 0 and rng.random() < r:
            v = 0

        brake_prob = float(P4)
        if int(front_id[i]) != -1 and gap_i <= S:
            brake_prob = max(brake_prob, 1.0 - float(P2))
        elif int(front_id[i]) != -1 and gap_i <= G:
            brake_prob = max(brake_prob, 1.0 - float(P3))

        if v > 0 and rng.random() < brake_prob:
            v -= 1

        if int(front_id[i]) != -1:
            v = min(v, gap_i)
        v = max(0, min(v, vmax_i))
        out_vel[i] = v

    out_vel = out_vel.astype(vel.dtype, copy=False)
    out_pos = pos.copy()
    for i in range(pos.shape[0]):
        if not alive[i]:
            continue
        out_pos[i] = (int(pos[i]) + int(out_vel[i])) % road_length
    return out_vel, out_pos


class _FakeRng:
    def __init__(self, seq):
        self._seq = list(seq)
        self.calls = 0

    def random(self):
        value = self._seq[self.calls]
        self.calls += 1
        return value


def test_import_smoke() -> None:
    assert callable(compute_longitudinal_velocities_kernel)
    assert callable(advance_positions_kernel)


def test_velocity_no_leader_accelerates_to_vmax() -> None:
    vel = np.array([0], dtype=np.int16)
    out = compute_longitudinal_velocities_kernel(
        vel, np.array([True]), np.array([False]), np.array([-1], dtype=np.int32), np.array([-1], dtype=np.int32),
        road_length=20, vmax_default=5, vmax_controlled=6, G=10, S=2, r=0.0, P2=1.0, P3=1.0, P4=0.0, rng=np.random.default_rng(0)
    )
    assert int(out[0]) == 1


def test_velocity_controlled_uses_controlled_vmax() -> None:
    vel = np.array([6], dtype=np.int16)
    out = compute_longitudinal_velocities_kernel(
        vel, np.array([True]), np.array([True]), np.array([-1], dtype=np.int32), np.array([-1], dtype=np.int32),
        road_length=20, vmax_default=5, vmax_controlled=7, G=10, S=2, r=0.0, P2=1.0, P3=1.0, P4=0.0, rng=np.random.default_rng(0)
    )
    assert int(out[0]) == 7


def test_velocity_gap_limits_speed() -> None:
    out = compute_longitudinal_velocities_kernel(
        np.array([5, 0], dtype=np.int16), np.array([True, True]), np.array([False, False]),
        np.array([1, -1], dtype=np.int32), np.array([2, -1], dtype=np.int32),
        road_length=20, vmax_default=5, vmax_controlled=6, G=10, S=2, r=0.0, P2=1.0, P3=1.0, P4=0.0, rng=np.random.default_rng(1)
    )
    assert int(out[0]) <= 2


def test_velocity_adjacent_front_forces_zero() -> None:
    out = compute_longitudinal_velocities_kernel(
        np.array([4, 0], dtype=np.int16), np.array([True, True]), np.array([False, False]),
        np.array([1, -1], dtype=np.int32), np.array([0, -1], dtype=np.int32),
        road_length=10, vmax_default=5, vmax_controlled=6, G=10, S=2, r=0.0, P2=1.0, P3=1.0, P4=0.0, rng=np.random.default_rng(1)
    )
    assert int(out[0]) == 0


def test_inactive_vehicle_velocity_unchanged() -> None:
    vel = np.array([1, 4], dtype=np.int16)
    out = compute_longitudinal_velocities_kernel(
        vel, np.array([True, False]), np.array([False, False]), np.array([-1, -1], dtype=np.int32), np.array([-1, -1], dtype=np.int32),
        road_length=10, vmax_default=5, vmax_controlled=6, G=10, S=2, r=0.0, P2=1.0, P3=1.0, P4=0.0, rng=np.random.default_rng(1)
    )
    assert int(out[1]) == 4


def test_position_kernel_wrap_inactive_and_no_mutation() -> None:
    pos = np.array([9, 5], dtype=np.int32)
    vel = np.array([2, 3], dtype=np.int16)
    alive = np.array([True, False])
    pos0, vel0, alive0 = pos.copy(), vel.copy(), alive.copy()
    out = advance_positions_kernel(pos, vel, alive, road_length=10)
    assert int(out[0]) == 1
    assert int(out[1]) == 5
    np.testing.assert_array_equal(pos, pos0)
    np.testing.assert_array_equal(vel, vel0)
    np.testing.assert_array_equal(alive, alive0)


def test_dtype_preservation_and_no_input_mutation() -> None:
    vel = np.array([0, 1], dtype=np.int16)
    pos = np.array([0, 1], dtype=np.int32)
    alive = np.array([True, True], dtype=np.bool_)
    controlled = np.array([False, True], dtype=np.bool_)
    front_id = np.array([-1, 0], dtype=np.int32)
    front_gap = np.array([-1, 3], dtype=np.int32)
    snapshots = [a.copy() for a in (vel, pos, alive, controlled, front_id, front_gap)]

    new_vel = compute_longitudinal_velocities_kernel(
        vel, alive, controlled, front_id, front_gap,
        road_length=20, vmax_default=5, vmax_controlled=6, G=10, S=2, r=0.0, P2=1.0, P3=1.0, P4=0.0, rng=np.random.default_rng(1)
    )
    new_pos = advance_positions_kernel(pos, new_vel, alive, road_length=20)

    assert new_vel.dtype == vel.dtype
    assert new_pos.dtype == pos.dtype
    for arr, snap in zip((vel, pos, alive, controlled, front_id, front_gap), snapshots):
        np.testing.assert_array_equal(arr, snap)


def test_rng_branch_draw_order_regression() -> None:
    vel = np.array([0, 1, 0], dtype=np.int16)
    alive = np.array([True, True, True], dtype=np.bool_)
    controlled = np.array([False, False, False], dtype=np.bool_)
    front_id = np.array([-1, -1, -1], dtype=np.int32)
    front_gap = np.array([-1, -1, -1], dtype=np.int32)
    fake = _FakeRng([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

    out = compute_longitudinal_velocities_kernel(
        vel, alive, controlled, front_id, front_gap,
        road_length=20, vmax_default=5, vmax_controlled=6, G=10, S=2, r=0.5, P2=1.0, P3=1.0, P4=0.5, rng=fake
    )
    assert fake.calls == 3
    np.testing.assert_array_equal(out, np.array([0, 2, 0], dtype=np.int16))


def test_kernel_strict_equivalence_to_legacy_loop() -> None:
    for num_lanes, road_length in [(1, 20), (2, 30), (4, 60)]:
        topology = RingTopology(num_lanes=num_lanes, length=road_length)
        params = SimulationParams(num_lanes=num_lanes, road_length=road_length)
        for density in [0.05, 0.2, 0.5, 1.0]:
            for seed in [0, 1, 2]:
                state = make_uniform_random_state(num_lanes=num_lanes, road_length=road_length, density=density, seed=seed)
                occupancy = build_occupancy(state, params)
                lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
                front_id, _, front_gap, _ = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

                rng_a = np.random.default_rng(seed + 1000)
                rng_b = np.random.default_rng(seed + 1000)

                exp_vel, exp_pos = _legacy_compute_longitudinal_arrays(
                    state.vel, state.pos, state.alive, state.controlled, front_id, front_gap,
                    road_length=params.road_length, vmax_default=params.vmax_default, vmax_controlled=params.vmax_controlled,
                    G=params.G, S=params.S, r=params.r, P2=params.P2, P3=params.P3, P4=params.P4, rng=rng_a,
                )
                got_vel = compute_longitudinal_velocities_kernel(
                    state.vel, state.alive, state.controlled, front_id, front_gap,
                    road_length=params.road_length, vmax_default=params.vmax_default, vmax_controlled=params.vmax_controlled,
                    G=params.G, S=params.S, r=params.r, P2=params.P2, P3=params.P3, P4=params.P4, rng=rng_b,
                )
                got_pos = advance_positions_kernel(state.pos, got_vel, state.alive, road_length=params.road_length)

                np.testing.assert_array_equal(got_vel, exp_vel)
                np.testing.assert_array_equal(got_pos, exp_pos)
                assert rng_a.random() == rng_b.random()

