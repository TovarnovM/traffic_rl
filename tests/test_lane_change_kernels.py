import numpy as np

from snfs_traffic.core import (
    SimulationParams,
    build_lane_order,
    build_occupancy,
    compute_neighbors,
    step_lane_change_reference,
)
from snfs_traffic.core.indexing import MISSING_INDEX
from snfs_traffic.core.lane_change_kernels import (
    apply_lane_changes_kernel,
    collect_lane_change_proposals_kernel,
    eligible_target_lanes_for_vehicle_kernel,
    resolve_lane_change_conflicts_kernel,
    target_lane_neighbors_at_pos_kernel,
)
from snfs_traffic.scenarios import make_uniform_random_state
from snfs_traffic.topology import RingTopology


class _FakeRng:
    def __init__(self, random_values=(), integer_values=()):
        self._random_values = list(random_values)
        self._integer_values = list(integer_values)
        self.calls = []

    def random(self):
        value = self._random_values.pop(0)
        self.calls.append(("random", tuple()))
        return value

    def integers(self, low, high=None):
        value = self._integer_values.pop(0)
        self.calls.append(("integers", (low, high)))
        return value


def _legacy_step_lane_change_arrays(lane, pos, vel, alive, controlled, changed_lane, last_lane_delta, occupancy, lane_order, lane_counts, front_id, front_gap, *, num_lanes, road_length, vmax_default, vmax_controlled, p_lane_change, rng):
    def _neighbors(target_lane, candidate_pos):
        count = int(lane_counts[target_lane])
        if count == 0:
            return MISSING_INDEX, MISSING_INDEX, -1, -1
        front_idx = MISSING_INDEX
        front_dist = None
        back_idx = MISSING_INDEX
        back_dist = None
        for rank in range(count):
            idx = int(lane_order[target_lane, rank])
            pos_j = int(pos[idx])
            dist_f = int((pos_j - candidate_pos) % road_length)
            dist_b = int((candidate_pos - pos_j) % road_length)
            if dist_f > 0 and (front_dist is None or dist_f < front_dist):
                front_dist = dist_f
                front_idx = idx
            if dist_b > 0 and (back_dist is None or dist_b < back_dist):
                back_dist = dist_b
                back_idx = idx
        if front_idx == MISSING_INDEX or back_idx == MISSING_INDEX:
            raise ValueError("target lane neighbor lookup failed")
        return front_idx, back_idx, int(front_dist - 1), int(back_dist - 1)

    proposals = {}
    for i in range(lane.shape[0]):
        if not alive[i]:
            continue
        lane_i, pos_i, v_i = int(lane[i]), int(pos[i]), int(vel[i])
        vmax_i = int(vmax_controlled if controlled[i] else vmax_default)
        if int(front_id[i]) == MISSING_INDEX:
            g_pf, v_p_front = road_length - 1, vmax_i
        else:
            g_pf, v_p_front = int(front_gap[i]), int(vel[int(front_id[i])])

        eligible = []
        for lane_delta in (-1, 1):
            target_lane = lane_i + lane_delta
            if target_lane < 0 or target_lane >= num_lanes:
                continue
            if int(occupancy[target_lane, pos_i]) != MISSING_INDEX:
                continue
            t_front, t_back, g_nf, g_nb = _neighbors(target_lane, pos_i)
            if t_front == MISSING_INDEX:
                g_nf, v_n_front, safety = road_length - 1, vmax_i, True
            else:
                v_n_front = int(vel[t_front])
                v_n_back = int(vel[t_back])
                safety = v_i > (v_n_back - g_nb)
            incentive = (g_nf + v_n_front > v_i) and (v_i >= g_pf + v_p_front)
            if incentive and safety:
                eligible.append(target_lane)

        if not eligible:
            continue
        target_lane = eligible[0]
        if len(eligible) > 1:
            target_lane = eligible[int(rng.integers(0, len(eligible)))]
        if float(rng.random()) >= p_lane_change:
            continue
        proposals.setdefault((target_lane, int(pos[i])), []).append(i)

    accepted = {}
    for (target_lane, _), candidates in proposals.items():
        winner = candidates[0] if len(candidates) == 1 else candidates[int(rng.integers(0, len(candidates)))]
        accepted[winner] = target_lane

    new_lane = lane.copy()
    new_changed = np.zeros_like(changed_lane, dtype=changed_lane.dtype)
    new_delta = np.zeros_like(last_lane_delta, dtype=last_lane_delta.dtype)
    for i, target_lane in accepted.items():
        old_lane = int(lane[i])
        new_lane[i] = np.asarray(target_lane, dtype=new_lane.dtype)
        new_changed[i] = True
        new_delta[i] = np.asarray(target_lane - old_lane, dtype=new_delta.dtype)
    return new_lane, new_changed, new_delta


def test_import_smoke():
    assert callable(target_lane_neighbors_at_pos_kernel)


def test_target_lane_neighbors_kernel_cases_and_no_mutation():
    pos = np.array([1, 8], dtype=np.int32)
    lane_order = np.array([[0, MISSING_INDEX], [1, MISSING_INDEX]], dtype=np.int32)
    lane_counts = np.array([1, 0], dtype=np.int32)
    s0 = (pos.copy(), lane_order.copy(), lane_counts.copy())
    out = target_lane_neighbors_at_pos_kernel(pos, lane_order, lane_counts, target_lane=1, candidate_pos=4, road_length=10)
    assert out == (MISSING_INDEX, MISSING_INDEX, -1, -1)

    lane_order2 = np.array([[0, 1, MISSING_INDEX]], dtype=np.int32)
    lane_counts2 = np.array([2], dtype=np.int32)
    pos2 = np.array([9, 2], dtype=np.int32)
    assert target_lane_neighbors_at_pos_kernel(pos2, lane_order2, lane_counts2, target_lane=0, candidate_pos=0, road_length=10) == (1, 0, 1, 0)
    assert target_lane_neighbors_at_pos_kernel(np.array([7], dtype=np.int32), np.array([[0]], dtype=np.int32), np.array([1], dtype=np.int32), target_lane=0, candidate_pos=2, road_length=10) == (0, 0, 4, 4)

    for a, b in zip((pos, lane_order, lane_counts), s0):
        np.testing.assert_array_equal(a, b)


def test_eligibility_semantics_cases():
    lane = np.array([0, 1, 2], dtype=np.int16)
    pos = np.array([5, 9, 1], dtype=np.int32)
    vel = np.array([2, 0, 0], dtype=np.int16)
    alive = np.array([True, True, True])
    controlled = np.array([False, False, False])
    occupancy = np.full((3, 20), MISSING_INDEX, dtype=np.int32)
    occupancy[0, 5] = 0
    occupancy[1, 9] = 1
    occupancy[2, 1] = 2
    lane_order = np.full((3, 3), MISSING_INDEX, dtype=np.int32)
    lane_order[0, 0], lane_order[1, 0], lane_order[2, 0] = 0, 1, 2
    lane_counts = np.array([1, 1, 1], dtype=np.int32)
    front_id = np.array([-1, -1, -1], dtype=np.int32)
    front_gap = np.array([-1, -1, -1], dtype=np.int32)

    assert eligible_target_lanes_for_vehicle_kernel(lane, pos, vel, alive, controlled, occupancy, lane_order, lane_counts, front_id, front_gap, vehicle_index=0, num_lanes=1, road_length=20, vmax_default=5, vmax_controlled=6) == []


def test_proposals_conflicts_apply_and_rng_order():
    lane = np.array([1, 1, 0], dtype=np.int16)
    pos = np.array([5, 5, 8], dtype=np.int32)
    vel = np.array([5, 4, 0], dtype=np.int16)
    alive = np.array([True, False, True])
    controlled = np.array([False, False, False])
    occupancy = np.full((3, 20), MISSING_INDEX, dtype=np.int32)
    occupancy[1, 5] = 0
    occupancy[0, 8] = 2
    lane_order = np.full((3, 3), MISSING_INDEX, dtype=np.int32)
    lane_order[0, 0], lane_order[1, 0] = 2, 0
    lane_counts = np.array([1, 1, 0], dtype=np.int32)
    front_id = np.array([-1, -1, -1], dtype=np.int32)
    front_gap = np.array([-1, -1, -1], dtype=np.int32)

    lane = np.array([1, 1], dtype=np.int16)
    pos = np.array([5, 6], dtype=np.int32)
    vel = np.array([5, 0], dtype=np.int16)
    alive = np.array([True, True])
    controlled = np.array([False, False])
    occupancy = np.full((3, 20), MISSING_INDEX, dtype=np.int32)
    occupancy[1, 5] = 0
    occupancy[1, 6] = 1
    lane_order = np.full((3, 2), MISSING_INDEX, dtype=np.int32)
    lane_order[1, 0] = 0
    lane_order[1, 1] = 1
    lane_counts = np.array([0, 2, 0], dtype=np.int32)
    front_id = np.array([1, 0], dtype=np.int32)
    front_gap = np.array([0, 18], dtype=np.int32)

    fake = _FakeRng(random_values=[0.1], integer_values=[1])
    proposals = collect_lane_change_proposals_kernel(lane, pos, vel, alive, controlled, occupancy, lane_order, lane_counts, front_id, front_gap, num_lanes=3, road_length=20, vmax_default=5, vmax_controlled=6, p_lane_change=1.0, rng=fake)
    assert fake.calls[0][0] == "integers"
    assert fake.calls[1][0] == "random"
    assert proposals == {(2, 5): [0]}

    fake_zero = _FakeRng(random_values=[0.0], integer_values=[0])
    p0 = collect_lane_change_proposals_kernel(lane, pos, vel, alive, controlled, occupancy, lane_order, lane_counts, front_id, front_gap, num_lanes=3, road_length=20, vmax_default=5, vmax_controlled=6, p_lane_change=0.0, rng=fake_zero)
    assert p0 == {}
    assert fake_zero.calls[-1] == ("random", tuple())

    accepted = resolve_lane_change_conflicts_kernel({(2, 5): [3, 1], (0, 1): [2]}, _FakeRng(integer_values=[1]))
    assert accepted == {1: 2, 2: 0}

    changed = np.array([True, True, True], dtype=np.bool_)
    delta = np.array([9, 9, 9], dtype=np.int8)
    l0, c0, d0 = lane.copy(), changed.copy(), delta.copy()
    new_lane, new_changed, new_delta = apply_lane_changes_kernel(lane, changed, delta, {0: 2})
    assert new_lane.dtype == lane.dtype and new_changed.dtype == changed.dtype and new_delta.dtype == delta.dtype
    assert bool(new_changed[0]) and int(new_delta[0]) == 1
    assert not bool(new_changed[1]) and int(new_delta[1]) == 0
    np.testing.assert_array_equal(lane, l0)
    np.testing.assert_array_equal(changed, c0)
    np.testing.assert_array_equal(delta, d0)


def test_kernel_and_public_reference_equivalence_to_legacy():
    for num_lanes, road_length in [(1, 20), (2, 30), (4, 60)]:
        topology = RingTopology(num_lanes=num_lanes, length=road_length)
        for density in [0.05, 0.2, 0.5, 1.0]:
            for p_lane_change in [0.0, 0.5, 1.0]:
                params = SimulationParams(num_lanes=num_lanes, road_length=road_length, p_lane_change=p_lane_change)
                for seed in [0, 1, 2]:
                    state = make_uniform_random_state(num_lanes=num_lanes, road_length=road_length, density=density, seed=seed)
                    occ = build_occupancy(state, params)
                    lane_order, lane_counts, lane_rank = build_lane_order(occ, n_vehicles=state.n_vehicles)
                    front_id, _, front_gap, _ = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

                    rng_a = np.random.default_rng(seed + 3000)
                    rng_b = np.random.default_rng(seed + 3000)
                    rng_c = np.random.default_rng(seed + 3000)
                    rng_d = np.random.default_rng(seed + 3000)

                    exp_lane, exp_changed, exp_delta = _legacy_step_lane_change_arrays(
                        state.lane, state.pos, state.vel, state.alive, state.controlled, state.changed_lane, state.last_lane_delta,
                        occ, lane_order, lane_counts, front_id, front_gap,
                        num_lanes=num_lanes, road_length=road_length,
                        vmax_default=params.vmax_default, vmax_controlled=params.vmax_controlled,
                        p_lane_change=p_lane_change, rng=rng_a,
                    )
                    proposals = collect_lane_change_proposals_kernel(
                        state.lane, state.pos, state.vel, state.alive, state.controlled,
                        occ, lane_order, lane_counts, front_id, front_gap,
                        num_lanes=num_lanes, road_length=road_length,
                        vmax_default=params.vmax_default, vmax_controlled=params.vmax_controlled,
                        p_lane_change=p_lane_change, rng=rng_b,
                    )
                    accepted = resolve_lane_change_conflicts_kernel(proposals, rng_b)
                    got_lane, got_changed, got_delta = apply_lane_changes_kernel(state.lane, state.changed_lane, state.last_lane_delta, accepted)

                    np.testing.assert_array_equal(exp_lane, got_lane)
                    np.testing.assert_array_equal(exp_changed, got_changed)
                    np.testing.assert_array_equal(exp_delta, got_delta)
                    assert rng_a.random() == rng_b.random()

                    ref = step_lane_change_reference(state, params, topology, rng_c)
                    np.testing.assert_array_equal(ref.lane, exp_lane)
                    np.testing.assert_array_equal(ref.changed_lane, exp_changed)
                    np.testing.assert_array_equal(ref.last_lane_delta, exp_delta)
                    _legacy_step_lane_change_arrays(
                        state.lane, state.pos, state.vel, state.alive, state.controlled, state.changed_lane, state.last_lane_delta,
                        occ, lane_order, lane_counts, front_id, front_gap,
                        num_lanes=num_lanes, road_length=road_length,
                        vmax_default=params.vmax_default, vmax_controlled=params.vmax_controlled,
                        p_lane_change=p_lane_change, rng=rng_d,
                    )
                    assert rng_d.random() == rng_c.random()
