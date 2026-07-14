from __future__ import annotations

import numpy as np
import pytest

from snfs_traffic.core import (
    PRIORITY_VEH_TYPE,
    SimulationParams,
    empty_state,
    high_speed_vehicle_mask,
    priority_vehicle_mask,
    validate_state,
)
from snfs_traffic.backends import get_backend
from snfs_traffic.topology import RingTopology
from snfs_traffic.scenarios import (
    AV_VEH_TYPE,
    PRIORITY_BEHAVIOR_ID,
    PRIORITY_YIELD_BEHAVIOR_ID,
    PriorityVehicleConfig,
    assign_priority_vehicles,
    select_yielding_vehicle_ids,
)


def _placement_state():
    state = empty_state(6)
    state.lane[:] = np.asarray([0, 0, 0, 1, 1, 1], dtype=state.lane.dtype)
    state.pos[:] = np.asarray([0, 16, 47, 2, 25, 70], dtype=state.pos.dtype)
    return state


def test_priority_role_uses_high_speed_limit_without_becoming_controlled():
    params = SimulationParams(
        num_lanes=2, road_length=100, vmax_default=5, vmax_controlled=6
    )
    state = empty_state(1)
    state.veh_type[0] = PRIORITY_VEH_TYPE
    state.vel[0] = 6

    validate_state(state, params)

    assert priority_vehicle_mask(state).tolist() == [True]
    assert high_speed_vehicle_mask(state).tolist() == [True]
    assert not bool(state.controlled[0])


def test_non_priority_uncontrolled_vehicle_cannot_exceed_default_vmax():
    params = SimulationParams(
        num_lanes=2, road_length=100, vmax_default=5, vmax_controlled=6
    )
    state = empty_state(1)
    state.vel[0] = 6

    with pytest.raises(ValueError, match="uncontrolled"):
        validate_state(state, params)


def test_priority_vehicle_reaches_vmax_six_in_reference_and_optimized_backends():
    params = SimulationParams(
        num_lanes=2,
        road_length=100,
        vmax_default=5,
        vmax_controlled=6,
        p_lane_change=0.0,
        P1=1.0,
        P2=1.0,
        P3=1.0,
        P4=1.0,
    )
    topology = RingTopology(num_lanes=2, length=100)
    state = empty_state(1)
    state.veh_type[0] = PRIORITY_VEH_TYPE
    state.vel[0] = 5

    reference = get_backend("reference").step(
        state, params, topology, np.random.default_rng(10)
    )
    optimized = get_backend("optimized").step(
        state, params, topology, np.random.default_rng(10)
    )

    assert int(reference.vel[0]) == 6
    assert int(optimized.vel[0]) == 6
    np.testing.assert_array_equal(optimized.pos, reference.pos)


def test_convoy_assignment_finds_requested_empty_gap_reproducibly():
    config = PriorityVehicleConfig(count=2, placement="convoy", target_gap=15, seed=7)

    first = assign_priority_vehicles(_placement_state(), config, road_length=100)
    second = assign_priority_vehicles(_placement_state(), config, road_length=100)

    assert first.vehicle_ids == (0, 1)
    assert first.vehicle_ids == second.vehicle_ids
    assert first.lane == 0
    assert first.actual_gaps == (15,)
    assert np.all(first.state.veh_type[[0, 1]] == PRIORITY_VEH_TYPE)
    assert np.all(first.state.behavior_id[[0, 1]] == PRIORITY_BEHAVIOR_ID)
    assert not np.any(first.state.controlled[[0, 1]])


def test_matched_yielding_selection_excludes_priority_and_tags_avs():
    assignment = assign_priority_vehicles(
        _placement_state(), PriorityVehicleConfig(count=1, seed=1), road_length=100
    )

    selected_state, vehicle_ids = select_yielding_vehicle_ids(
        assignment.state, fraction=0.4, seed=11
    )

    assert len(vehicle_ids) == 2
    assert set(vehicle_ids).isdisjoint(assignment.vehicle_ids)
    selected_indices = [
        int(np.flatnonzero(selected_state.vehicle_id == vehicle_id)[0])
        for vehicle_id in vehicle_ids
    ]
    assert np.all(selected_state.veh_type[selected_indices] == AV_VEH_TYPE)
    assert np.all(
        selected_state.behavior_id[selected_indices] == PRIORITY_YIELD_BEHAVIOR_ID
    )
    assert not np.any(selected_state.controlled[selected_indices])


def test_matched_penetration_sets_are_nested_when_seed_is_shared():
    assignment = assign_priority_vehicles(
        _placement_state(), PriorityVehicleConfig(count=1, seed=1), road_length=100
    )

    _state_small, ids_small = select_yielding_vehicle_ids(
        assignment.state, fraction=0.2, seed=99
    )
    _state_large, ids_large = select_yielding_vehicle_ids(
        assignment.state, fraction=0.8, seed=99
    )

    assert set(ids_small) < set(ids_large)
