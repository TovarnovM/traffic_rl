"""Source-only W5 regressions; parent runs these in the DevContainer."""

import copy

import numpy as np
import pytest

from physica_mc import diagnostics
from snfs_traffic import control
from snfs_traffic.core import SimulationParams, TrafficState
from snfs_traffic.core.indexing import build_body_occupancy, build_lane_order, build_occupancy
from snfs_traffic.core.indexing_kernels import compute_cumulative_forward_gap_kernel, compute_neighbors_kernel
from snfs_traffic.core.lane_change_kernels import eligible_target_lanes_for_vehicle_kernel
from snfs_traffic.core.roles import high_speed_vehicle_mask
from snfs_traffic.topology import RingTopology


def state(lanes, positions, speeds):
    n = len(lanes)
    return TrafficState(
        vehicle_id=np.arange(n, dtype=np.int32),
        lane=np.array(lanes, dtype=np.int16),
        pos=np.array(positions, dtype=np.int32),
        vel=np.array(speeds, dtype=np.int16),
        length=np.ones(n, dtype=np.int16),
        veh_type=np.array([3] + [1] * (n - 1), dtype=np.int16),
        behavior_id=np.zeros(n, dtype=np.int16),
        alive=np.ones(n, dtype=bool),
        last_lane_delta=np.zeros(n, dtype=np.int8),
        changed_lane=np.zeros(n, dtype=bool),
        controlled=np.zeros(n, dtype=bool),
    )


def owning_eligible(s, params):
    order, counts, rank = build_lane_order(build_occupancy(s, params), n_vehicles=s.n_vehicles)
    front, _, gap, _ = compute_neighbors_kernel(
        s.lane, s.pos, s.alive, order, counts, rank, road_length=params.road_length
    )
    return eligible_target_lanes_for_vehicle_kernel(
        s.lane, s.pos, s.vel, s.length, s.alive, high_speed_vehicle_mask(s),
        build_body_occupancy(s, params), order, counts, front, gap,
        vehicle_index=0, num_lanes=params.num_lanes, road_length=params.road_length,
        vmax_default=params.vmax_default, vmax_controlled=params.vmax_controlled,
    )


@pytest.mark.parametrize("lanes,positions,speeds,expected", [
    ([1, 1], [10, 12], [4, 0], (1, 1, 2)),  # empty target lanes
    ([0, 0], [10, 12], [4, 0], (0, 1, 1)),  # missing left lane
    ([2, 2], [10, 12], [4, 0], (1, 0, 1)),  # missing right lane
    ([1, 1], [10, 30], [4, 0], (0, 0, 0)),  # incentive false
    ([1, 1, 0], [10, 12, 9], [4, 0, 5], (0, 1, 1)),  # rear safety false
    ([1, 1, 0], [10, 12, 10], [4, 0, 2], (0, 1, 1)),  # body occupied
    ([1], [10], [6], (0, 0, 0)),  # absent current leader
])
def test_native_rule_exact_agreement_and_no_mutation(lanes, positions, speeds, expected):
    params = SimulationParams(num_lanes=3, road_length=100)
    before = state(lanes, positions, speeds)
    saved = before.copy()
    assert diagnostics.native_eligibility(before, params, 0) == expected
    eligible = owning_eligible(before, params)
    assert expected[:2] == (int(lanes[0] - 1 in eligible), int(lanes[0] + 1 in eligible))
    for name in before.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(before, name), getattr(saved, name))


def test_native_rule_uses_priority_high_speed_mask(monkeypatch):
    before = state([1, 1], [10, 12], [6, 0])
    real = diagnostics.eligible_target_lanes_for_vehicle_kernel
    seen = []

    def capture(*args, **kwargs):
        seen.append(args[5].copy())
        return real(*args, **kwargs)

    monkeypatch.setattr(diagnostics, "eligible_target_lanes_for_vehicle_kernel", capture)
    diagnostics.native_eligibility(before, SimulationParams(num_lanes=3, road_length=100), 0)
    np.testing.assert_array_equal(seen[0], [True, False])
    assert not before.controlled.any()


@pytest.mark.parametrize("opening", [True, False])
@pytest.mark.parametrize("optimized", [False, True])
def test_applied_av_move_is_not_native_collector_geometry(opening, optimized, monkeypatch):
    params = SimulationParams(num_lanes=3, road_length=100, p_lane_change=0.0)
    before = state([2, 2, 1 if opening else 0], [10, 12, 10], [4, 0, 2])
    saved = before.copy()
    rng = np.random.default_rng(73)
    captured = []
    # Capture the exact arrays passed by the owner, without any extra RNG call.
    for name in ("collect_lane_change_proposals_kernel", "collect_lane_change_proposals_numba"):
        real = getattr(control, name)

        def capture(*args, _real=real, **kwargs):
            captured.append((args[0] if args else kwargs["lane"]).copy())
            return _real(*args, **kwargs)

        monkeypatch.setattr(control, name, capture)
    step = control.step_with_lateral_overrides_optimized if optimized else control.step_with_lateral_overrides_reference
    output = step(before, params, RingTopology(num_lanes=3, length=100), rng, {2: -1 if opening else 1})
    after, feedback = output[:2]
    assert feedback.applied.tolist() == [True]
    np.testing.assert_array_equal(captured[0], before.lane)
    counterfactual = before.copy()
    counterfactual.lane[2] += feedback.applied_lane_delta[0]
    assert diagnostics.native_eligibility(counterfactual, params, 0)[0] == int(opening)
    rng_saved = copy.deepcopy(rng.bit_generator.state)
    after_saved = after.copy()
    observed = diagnostics.observe_transition(before, after, params, 0)
    assert observed["pre_pv_native_eligible_left"] == int(not opening)
    assert observed["native_phase_pv_native_eligible_left"] == int(not opening)
    assert rng.bit_generator.state == rng_saved
    for name in before.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(before, name), getattr(saved, name))
        np.testing.assert_array_equal(getattr(after, name), getattr(after_saved, name))


def test_longitudinal_reconstruction_uses_all_final_lanes_and_owning_bounds():
    params = SimulationParams(num_lanes=3, road_length=100, S=2)
    before = state([1, 1, 0], [10, 12, 20], [4, 0, 2])
    after = before.copy()
    after.lane[:] = [1, 0, 1]
    after.vel[:] = [3, 1, 2]
    after.pos[:] = (before.pos + after.vel) % 100
    values = diagnostics.observe_transition(before, after, params, 0)
    assert values["long_pv_leader_gap"] == 9
    assert values["long_pv_leader_speed"] == 2
    assert values["long_pv_leader_final_speed"] == 2
    assert values["long_pv_acceleration_gate"] == 0
    assert values["long_pv_v1"] == 4
    assert values["long_pv_eq9_slack"] == 8
    assert values["long_pv_eq9_tight"] == 0
    reconstructed = before.copy()
    reconstructed.lane = after.lane.copy()
    order, counts, rank = build_lane_order(build_occupancy(reconstructed, params), n_vehicles=3)
    for phase, positions in (("cur", before.pos), ("prev", (before.pos - before.vel) % 100)):
        for suffix, k in (("s1", 1), ("sS", params.S)):
            assert values[f"long_pv_{phase}_gap_{suffix}"] == compute_cumulative_forward_gap_kernel(
                0, k, positions, before.length, order, counts, rank, road_length=100
            )


@pytest.mark.parametrize("S", [1, 2, 4])
def test_singleton_longitudinal_bounds_and_missing_leader(S):
    before = state([1], [0], [5])
    after = before.copy()
    after.vel[0], after.pos[0] = 6, 6
    values = diagnostics.observe_transition(before, after, SimulationParams(num_lanes=3, road_length=100, S=S), 0)
    for suffix in ("leader_gap", "leader_type", "leader_speed", "leader_final_speed", "eq9_slack", "eq9_tight"):
        assert values[f"long_pv_{suffix}"] == -1
    assert values["long_pv_acceleration_gate"] == 1
    assert values["long_pv_v1"] == 6
    for phase in ("prev", "cur"):
        assert values[f"long_pv_{phase}_gap_s1"] == 99
        assert values[f"long_pv_{phase}_gap_sS"] == S * 99


def test_eq9_tightness_uses_final_leader_speed_not_initial_speed():
    before = state([1, 1], [10, 12], [4, 0])
    after = before.copy()
    after.vel[:] = [2, 1]
    after.pos[:] += after.vel
    values = diagnostics.observe_transition(before, after, SimulationParams(num_lanes=3, road_length=100), 0)
    assert values["long_pv_leader_speed"] == 0
    assert values["long_pv_leader_final_speed"] == 1
    assert values["long_pv_eq9_slack"] == 0
    assert values["long_pv_eq9_tight"] == 1
