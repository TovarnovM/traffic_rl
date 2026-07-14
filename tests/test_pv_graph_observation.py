from __future__ import annotations

import numpy as np
import pytest

from snfs_traffic.core import PRIORITY_VEH_TYPE, SimulationParams, empty_state
from snfs_traffic.observations import PvGraphConfig, build_pv_graph_observation
from snfs_traffic.scenarios import AV_VEH_TYPE, HDV_VEH_TYPE
from snfs_traffic.topology import RingTopology


def _graph_state():
    state = empty_state(6)
    state.lane[:] = np.asarray([1, 1, 0, 3, 2, 2], dtype=state.lane.dtype)
    state.pos[:] = np.asarray([98, 2, 90, 50, 1, 30], dtype=state.pos.dtype)
    state.vel[:] = np.asarray([4, 3, 2, 2, 1, 3], dtype=state.vel.dtype)
    state.veh_type[:] = np.asarray(
        [
            PRIORITY_VEH_TYPE,
            AV_VEH_TYPE,
            AV_VEH_TYPE,
            AV_VEH_TYPE,
            HDV_VEH_TYPE,
            AV_VEH_TYPE,
        ],
        dtype=state.veh_type.dtype,
    )
    return state


def test_graph_selects_av_across_periodic_boundary_and_sorts_by_relative_position():
    params = SimulationParams(num_lanes=4, road_length=100)
    topology = RingTopology(num_lanes=4, length=100)
    graph = build_pv_graph_observation(
        _graph_state(),
        params,
        topology,
        priority_vehicle_id=0,
        config=PvGraphConfig(
            front_distance=10,
            back_distance=10,
            sensor_distance=20,
            cooldown_steps=5,
        ),
    )

    assert graph.active_count == 2
    assert graph.vehicle_id[:2].tolist() == [2, 1]
    assert graph.node_features[0, 0] == pytest.approx(-0.8)
    assert graph.node_features[1, 0] == pytest.approx(0.4)
    assert graph.global_features[2] == pytest.approx(6 / 400)


def test_padding_only_allows_stay_and_is_not_exposed_as_active():
    params = SimulationParams(num_lanes=4, road_length=100)
    topology = RingTopology(num_lanes=4, length=100)
    graph = build_pv_graph_observation(
        _graph_state(),
        params,
        topology,
        priority_vehicle_id=0,
        config=PvGraphConfig(
            front_distance=10,
            back_distance=10,
            sensor_distance=20,
        ),
    )

    np.testing.assert_array_equal(graph.node_mask[:2], np.ones(2, dtype=np.int8))
    np.testing.assert_array_equal(graph.node_mask[2:], np.zeros_like(graph.node_mask[2:]))
    np.testing.assert_array_equal(
        graph.action_mask[2:],
        np.tile(np.asarray([0, 1, 0], dtype=np.int8), (graph.action_mask.shape[0] - 2, 1)),
    )
    assert np.all(graph.vehicle_id[2:] == -1)


def test_cooldown_masks_both_lane_changes_for_physical_vehicle_id():
    params = SimulationParams(num_lanes=4, road_length=100)
    topology = RingTopology(num_lanes=4, length=100)
    graph = build_pv_graph_observation(
        _graph_state(),
        params,
        topology,
        priority_vehicle_id=0,
        config=PvGraphConfig(
            front_distance=10,
            back_distance=10,
            sensor_distance=20,
            cooldown_steps=5,
        ),
        cooldown_by_vehicle_id={1: 3},
    )

    slot = int(np.flatnonzero(graph.vehicle_id == 1)[0])
    np.testing.assert_array_equal(graph.action_mask[slot], [0, 1, 0])
    assert graph.node_features[slot, 6] == pytest.approx(3 / 5)


def test_nearest_hdv_is_encoded_but_not_added_as_graph_node():
    params = SimulationParams(num_lanes=4, road_length=100)
    topology = RingTopology(num_lanes=4, length=100)
    graph = build_pv_graph_observation(
        _graph_state(),
        params,
        topology,
        priority_vehicle_id=0,
        config=PvGraphConfig(
            front_distance=10,
            back_distance=10,
            sensor_distance=20,
        ),
    )

    av_slot = int(np.flatnonzero(graph.vehicle_id == 1)[0])
    # The nearest HDV is vehicle 4, one cell behind AV 1 across another lane.
    hdv_feature_start = 7 + 3 * 11
    assert graph.node_features[av_slot, hdv_feature_start + 4] == 1.0
    assert graph.node_features[av_slot, hdv_feature_start + 5] == pytest.approx(1 / 20)
    assert 4 not in graph.vehicle_id


def test_missing_priority_vehicle_is_rejected():
    params = SimulationParams(num_lanes=4, road_length=100)
    topology = RingTopology(num_lanes=4, length=100)

    with pytest.raises(ValueError, match="priority_vehicle_id"):
        build_pv_graph_observation(
            _graph_state(),
            params,
            topology,
            priority_vehicle_id=999,
            config=PvGraphConfig(sensor_distance=20),
        )
