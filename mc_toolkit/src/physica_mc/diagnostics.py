"""Passive W5 facts; no RNG, policy calls, or writes to simulator state.

Native collectors use pre-step geometry even after explicit moves have been
accepted/reserved. Longitudinal geometry uses all final lanes, before motion.
"""

import numpy as np

from snfs_traffic.core.indexing import build_body_occupancy, build_lane_order, build_occupancy
from snfs_traffic.core.indexing_kernels import (
    compute_cumulative_forward_gap_kernel,
    compute_neighbors_kernel,
)
from snfs_traffic.core.lane_change_kernels import eligible_target_lanes_for_vehicle_kernel
from snfs_traffic.core.roles import high_speed_vehicle_mask


COLUMNS = {
    "pre_pv_lane": ("int16", "PV lane before explicit/native lateral actions"),
}
for _phase in ("pre", "native_phase"):
    for _side in ("left", "right", "count"):
        COLUMNS[f"{_phase}_pv_native_eligible_{_side}"] = (
            "int8",
            f"{_phase}: pure native incentive+safety+geometry eligibility; "
            + ("number of eligible directions 0..2" if _side == "count" else "0/1; missing lane is 0")
            + "; excludes stochastic p gate, direction choice and reserved-target/conflict filtering",
        )
COLUMNS.update({
    "long_pv_lane": ("int16", "PV lane after all lateral moves, before longitudinal motion"),
    "long_pv_length": ("int32", "PV body length in cells"),
    "long_pv_speed_before": ("int8", "PV pre-step speed v0"),
    "long_pv_speed_after": ("int8", "PV actual final speed after Eq9"),
    "long_pv_leader_gap": ("int32", "empty-cell gap after all lateral moves at pre-step positions; -1 no leader"),
    "long_pv_leader_type": ("int8", "longitudinal leader veh_type; -1 no leader"),
    "long_pv_leader_speed": ("int8", "longitudinal leader pre-step speed; -1 no leader"),
    "long_pv_leader_final_speed": ("int8", "same leader actual final speed after Eq9; -1 no leader"),
    "long_pv_acceleration_gate": ("int8", "0/1: g>=G or v0<=leader v0; singleton uses g=L-length and leader v0=PV v0"),
    "long_pv_v1": ("int8", "deterministic acceleration-stage speed, before stochastic s/q/braking"),
    "long_pv_prev_gap_s1": ("int64", "owning cumulative gap k=1 at (pre_pos-pre_vel)%L using post-lateral lane order"),
    "long_pv_prev_gap_sS": ("int64", "owning cumulative gap k=S at (pre_pos-pre_vel)%L using post-lateral lane order"),
    "long_pv_cur_gap_s1": ("int64", "owning cumulative gap k=1 at pre_pos using post-lateral lane order"),
    "long_pv_cur_gap_sS": ("int64", "owning cumulative gap k=S at pre_pos using post-lateral lane order"),
    "long_pv_eq9_slack": ("int32", "g+leader final speed-PV final speed; -1 no leader; tightness does not prove causal speed reduction"),
    "long_pv_eq9_tight": ("int8", "1 iff Eq9 slack=0, else 0; -1 no leader"),
})
COLUMNS = {name: {"dtype": dtype, "description": description}
           for name, (dtype, description) in COLUMNS.items()}


def native_eligibility(state, params, pv_index):
    """Return left/right/count from the owning pure native rule."""
    occupancy = build_occupancy(state, params)
    order, counts, rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front, _, gap, _ = compute_neighbors_kernel(
        state.lane, state.pos, state.alive, order, counts, rank,
        road_length=params.road_length,
    )
    eligible = eligible_target_lanes_for_vehicle_kernel(
        state.lane, state.pos, state.vel, state.length, state.alive,
        high_speed_vehicle_mask(state), build_body_occupancy(state, params),
        order, counts, front, gap, vehicle_index=pv_index,
        num_lanes=params.num_lanes, road_length=params.road_length,
        vmax_default=params.vmax_default, vmax_controlled=params.vmax_controlled,
    )
    lane = int(state.lane[pv_index])
    left, right = int(lane - 1 in eligible), int(lane + 1 in eligible)
    return left, right, left + right


def observe_transition(before, after, params, pv_index):
    """Reconstruct only observable phases on private copies.

    control._step_with_lateral_overrides passes ORIGINAL state/index arrays to
    both native collectors; accepted explicit moves are only reservations until
    apply_lane_changes_kernel merges both sets. Thus native eligibility equals
    pre-action eligibility for the PV, which this protocol never overrides.
    """
    pre = before.copy()
    values = {"pre_pv_lane": int(pre.lane[pv_index])}
    eligible = native_eligibility(pre, params, pv_index)
    for phase in ("pre", "native_phase"):
        for side, value in zip(("left", "right", "count"), eligible):
            values[f"{phase}_pv_native_eligible_{side}"] = value

    pre.lane = after.lane.copy()
    order, counts, rank = build_lane_order(
        build_occupancy(pre, params), n_vehicles=pre.n_vehicles
    )
    i = pv_index
    lane, length, v0 = int(pre.lane[i]), int(pre.length[i]), int(pre.vel[i])
    leader = int(order[lane, (int(rank[i]) + 1) % int(counts[lane])]) if counts[lane] > 1 else -1
    g = (int(pre.pos[leader]) - int(pre.pos[i]) - length) % params.road_length if leader >= 0 else params.road_length - length
    leader_v = int(pre.vel[leader]) if leader >= 0 else v0
    gate = int(g >= params.G or v0 <= leader_v)
    vmax = params.vmax_controlled if high_speed_vehicle_mask(pre)[i] else params.vmax_default
    final_v = int(after.vel[i])
    slack = g + int(after.vel[leader]) - final_v if leader >= 0 else -1
    values.update({
        "long_pv_lane": lane,
        "long_pv_length": length,
        "long_pv_speed_before": v0,
        "long_pv_speed_after": final_v,
        "long_pv_leader_gap": g if leader >= 0 else -1,
        "long_pv_leader_type": int(pre.veh_type[leader]) if leader >= 0 else -1,
        "long_pv_leader_speed": leader_v if leader >= 0 else -1,
        "long_pv_leader_final_speed": int(after.vel[leader]) if leader >= 0 else -1,
        "long_pv_acceleration_gate": gate,
        "long_pv_v1": min(vmax, v0 + 1) if gate else v0,
        "long_pv_eq9_slack": slack,
        "long_pv_eq9_tight": int(slack == 0) if leader >= 0 else -1,
    })
    previous = (pre.pos.astype(np.int64) - pre.vel.astype(np.int64)) % params.road_length
    for phase, positions in (("prev", previous), ("cur", pre.pos)):
        for suffix, k in (("s1", 1), ("sS", params.S)):
            values[f"long_pv_{phase}_gap_{suffix}"] = compute_cumulative_forward_gap_kernel(
                i, k, positions, pre.length, order, counts, rank,
                road_length=params.road_length,
            )
    return values
