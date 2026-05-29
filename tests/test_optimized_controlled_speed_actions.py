import numpy as np
import pytest

from snfs_traffic.backends import optimized_backend_available
from snfs_traffic.control import (
    LANE_STAY,
    ControlledVehicleAction,
    step_with_controlled_lateral_actions_optimized,
    step_with_controlled_lateral_actions_reference,
)
from snfs_traffic.core import SimulationParams, empty_state
from snfs_traffic.core.longitudinal_numba import NUMBA_AVAILABLE as LONG_NUMBA_AVAILABLE
from snfs_traffic.core.state import validate_state
from snfs_traffic.rl.episode import EpisodeConfig
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.simulator import TrafficSimulator
from snfs_traffic.topology import RingTopology

pytestmark = pytest.mark.filterwarnings("ignore:.*Numba.*")


def _params(**kwargs):
    defaults = dict(num_lanes=1, road_length=40, vmax_default=5, vmax_controlled=6, p_lane_change=0.0)
    defaults.update(kwargs)
    return SimulationParams(**defaults)


def _state(*, pos, vel, controlled, length=None, lane=None):
    state = empty_state(len(pos))
    state.lane[:] = 0 if lane is None else np.asarray(lane, dtype=state.lane.dtype)
    state.pos[:] = np.asarray(pos, dtype=state.pos.dtype)
    state.vel[:] = np.asarray(vel, dtype=state.vel.dtype)
    state.controlled[:] = np.asarray(controlled, dtype=state.controlled.dtype)
    if length is not None:
        state.length[:] = np.asarray(length, dtype=state.length.dtype)
    return state


def _compare_arrays(left, right):
    for name in ("lane", "pos", "vel", "alive", "controlled", "length"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)


def _compare_results(left, right):
    np.testing.assert_array_equal(left.vehicle_id, right.vehicle_id)
    np.testing.assert_array_equal(left.requested_lane_delta, right.requested_lane_delta)
    np.testing.assert_array_equal(left.applied_lane_delta, right.applied_lane_delta)
    np.testing.assert_array_equal(left.valid, right.valid)
    np.testing.assert_array_equal(left.applied, right.applied)
    assert left.rejection_reason == right.rejection_reason
    if left.requested_speed_delta is None or right.requested_speed_delta is None:
        assert left.requested_speed_delta is right.requested_speed_delta
        return
    np.testing.assert_array_equal(left.requested_speed_delta, right.requested_speed_delta)
    np.testing.assert_array_equal(left.desired_velocity, right.desired_velocity)
    np.testing.assert_array_equal(left.applied_velocity, right.applied_velocity)
    np.testing.assert_array_equal(left.speed_clipped_by_safety, right.speed_clipped_by_safety)


def _step_both(state, actions, params=None, seed=123):
    params = params or _params()
    topology = RingTopology(num_lanes=params.num_lanes, length=params.road_length)
    ref, ref_result = step_with_controlled_lateral_actions_reference(
        state, params, topology, np.random.default_rng(seed), actions
    )
    opt, opt_result, used_reference = step_with_controlled_lateral_actions_optimized(
        state, params, topology, np.random.default_rng(seed), actions
    )
    return ref, ref_result, opt, opt_result, used_reference


@pytest.mark.skipif(not LONG_NUMBA_AVAILABLE, reason="Numba optional dependency is not installed")
def test_optimized_backend_accepts_explicit_speed_delta_for_unit_length():
    params = _params()
    topology = RingTopology(num_lanes=params.num_lanes, length=params.road_length)
    state = _state(pos=(0, 20), vel=(2, 0), controlled=(True, False))

    out, result, used_reference = step_with_controlled_lateral_actions_optimized(
        state,
        params,
        topology,
        np.random.default_rng(123),
        {0: ControlledVehicleAction(LANE_STAY, 0)},
    )

    assert used_reference is False
    validate_state(out, params)
    assert int(result.requested_speed_delta[0]) == 0


@pytest.mark.skipif(not LONG_NUMBA_AVAILABLE, reason="Numba optional dependency is not installed")
@pytest.mark.parametrize(
    ("speed_delta", "expected_velocity"),
    [(-1, 2), (0, 3), (1, 4)],
)
def test_brake_keep_accelerate_equivalence(speed_delta, expected_velocity):
    state = _state(pos=(0, 25), vel=(3, 0), controlled=(True, False))

    ref, ref_result, opt, opt_result, used_reference = _step_both(
        state, {0: ControlledVehicleAction(LANE_STAY, speed_delta)}
    )

    assert used_reference is False
    _compare_arrays(ref, opt)
    _compare_results(ref_result, opt_result)
    assert int(opt.vel[0]) == expected_velocity
    assert int(opt_result.desired_velocity[0]) == expected_velocity
    assert bool(opt_result.speed_clipped_by_safety[0]) is False


@pytest.mark.skipif(not LONG_NUMBA_AVAILABLE, reason="Numba optional dependency is not installed")
def test_vmax_clipping_equivalence():
    params = _params(vmax_controlled=3)
    state = _state(pos=(0, 25), vel=(3, 0), controlled=(True, False))

    ref, ref_result, opt, opt_result, used_reference = _step_both(
        state, {0: ControlledVehicleAction(LANE_STAY, 1)}, params=params
    )

    assert used_reference is False
    _compare_arrays(ref, opt)
    _compare_results(ref_result, opt_result)
    assert int(opt_result.desired_velocity[0]) == params.vmax_controlled
    assert int(opt_result.applied_velocity[0]) == params.vmax_controlled


@pytest.mark.skipif(not LONG_NUMBA_AVAILABLE, reason="Numba optional dependency is not installed")
def test_safety_clipping_equivalence_and_avoids_overlap():
    params = _params(road_length=10, vmax_default=0, vmax_controlled=6)
    state = _state(pos=(0, 2), vel=(1, 0), controlled=(True, False))

    ref, ref_result, opt, opt_result, used_reference = _step_both(
        state, {0: ControlledVehicleAction(LANE_STAY, 1)}, params=params
    )

    assert used_reference is False
    _compare_arrays(ref, opt)
    _compare_results(ref_result, opt_result)
    assert int(opt.vel[0]) <= 1
    assert int(opt.pos[0]) != int(opt.pos[1])
    assert bool(opt_result.speed_clipped_by_safety[0]) is True


@pytest.mark.skipif(not LONG_NUMBA_AVAILABLE, reason="Numba optional dependency is not installed")
def test_mixed_legacy_and_explicit_actions_equivalence():
    params = _params(road_length=80, vmax_default=5, vmax_controlled=6)
    state = _state(
        pos=(0, 12, 30, 48, 65),
        vel=(2, 3, 3, 3, 0),
        controlled=(True, True, True, True, False),
    )
    actions = {
        0: LANE_STAY,
        1: ControlledVehicleAction(LANE_STAY, -1),
        2: ControlledVehicleAction(LANE_STAY, 0),
        3: ControlledVehicleAction(LANE_STAY, 1),
    }

    ref, ref_result, opt, opt_result, used_reference = _step_both(state, actions, params=params, seed=999)

    assert used_reference is False
    _compare_arrays(ref, opt)
    _compare_results(ref_result, opt_result)


@pytest.mark.skipif(not LONG_NUMBA_AVAILABLE, reason="Numba optional dependency is not installed")
def test_hdv_rng_parity_in_mixed_step():
    params = _params(road_length=80, vmax_default=5, vmax_controlled=6, p_lane_change=0.0)
    state = _state(
        pos=(0, 10, 25, 40, 60),
        vel=(2, 1, 3, 2, 4),
        controlled=(True, False, True, False, False),
    )
    actions = {
        0: ControlledVehicleAction(LANE_STAY, 1),
        2: ControlledVehicleAction(LANE_STAY, -1),
    }

    ref, ref_result, opt, opt_result, used_reference = _step_both(state, actions, params=params, seed=321)

    assert used_reference is False
    _compare_arrays(ref, opt)
    _compare_results(ref_result, opt_result)


@pytest.mark.skipif(not LONG_NUMBA_AVAILABLE, reason="Numba optional dependency is not installed")
def test_random_unit_length_equivalence():
    params = SimulationParams(num_lanes=3, road_length=50, p_lane_change=0.0)
    topology = RingTopology(num_lanes=params.num_lanes, length=params.road_length)
    for seed in range(10):
        state = make_uniform_random_state(
            num_lanes=params.num_lanes,
            road_length=params.road_length,
            density=0.2,
            seed=seed,
            vehicle_mix=VehicleMix(),
        )
        state.controlled[:] = False
        controlled_ids = [int(v) for v in state.vehicle_id[: min(5, state.n_vehicles)]]
        for vehicle_id in controlled_ids:
            state.controlled[vehicle_id] = True
        rng = np.random.default_rng(seed)
        actions = {
            vehicle_id: ControlledVehicleAction(LANE_STAY, int(rng.integers(-1, 2)))
            for vehicle_id in controlled_ids
        }

        ref, ref_result = step_with_controlled_lateral_actions_reference(
            state, params, topology, np.random.default_rng(seed + 100), actions
        )
        opt, opt_result, used_reference = step_with_controlled_lateral_actions_optimized(
            state, params, topology, np.random.default_rng(seed + 100), actions
        )

        assert used_reference is False
        _compare_arrays(ref, opt)
        _compare_results(ref_result, opt_result)


@pytest.mark.skipif(not optimized_backend_available(), reason="optimized backend optional dependencies are not installed")
def test_repeated_speed_control_env_rollout_equivalence():
    pytest.importorskip("gymnasium")
    from snfs_traffic.rl.speed_control_multiagent_env import SnfsTrafficSpeedControlMultiAgentEnv

    kwargs = dict(
        num_lanes=3,
        road_length=60,
        density=0.2,
        num_controlled=5,
        episode_config=EpisodeConfig(max_steps=20),
    )
    ref_env = SnfsTrafficSpeedControlMultiAgentEnv(backend="reference", **kwargs)
    opt_env = SnfsTrafficSpeedControlMultiAgentEnv(backend="optimized", **kwargs)
    ref_env.reset(seed=42)
    opt_env.reset(seed=42)

    for step in range(12):
        assert ref_env.agents == opt_env.agents
        actions = {
            agent_id: np.asarray([0, (step + int(agent_id.split("_")[1])) % 3], dtype=np.int64)
            for agent_id in ref_env.agents
        }
        ref_obs, ref_rewards, ref_terms, ref_truncs, ref_infos = ref_env.step(actions)
        opt_obs, opt_rewards, opt_terms, opt_truncs, opt_infos = opt_env.step(actions)

        assert ref_rewards == opt_rewards
        assert ref_terms == opt_terms
        assert ref_truncs == opt_truncs
        assert set(ref_obs) == set(opt_obs)
        assert set(ref_infos) == set(opt_infos)
        _compare_arrays(ref_env._sim.state, opt_env._sim.state)
        if ref_terms["__all__"] or ref_truncs["__all__"]:
            break


def test_length_greater_than_one_explicit_speed_falls_back_to_reference():
    params = _params(road_length=30)
    state = _state(pos=(0, 15), vel=(2, 0), controlled=(True, False), length=(2, 1))

    ref, ref_result, opt, opt_result, used_reference = _step_both(
        state, {0: ControlledVehicleAction(LANE_STAY, 1)}, params=params
    )

    assert used_reference is True
    _compare_arrays(ref, opt)
    _compare_results(ref_result, opt_result)


def test_old_env_action_spaces_remain_unchanged():
    pytest.importorskip("gymnasium")
    from snfs_traffic.rl.env import SnfsTrafficEnv
    from snfs_traffic.rl.multiagent_env import SnfsTrafficMultiAgentEnv
    from snfs_traffic.rl.speed_control_multiagent_env import SnfsTrafficSpeedControlMultiAgentEnv

    single = SnfsTrafficEnv(backend="reference")
    multi = SnfsTrafficMultiAgentEnv(backend="reference", num_controlled=2, density=0.3)
    speed = SnfsTrafficSpeedControlMultiAgentEnv(backend="reference", num_controlled=2, density=0.3)

    assert single.action_space.n == 3
    assert multi.single_agent_action_space.n == 3
    assert speed.single_agent_action_space.nvec.tolist() == [3, 3]
