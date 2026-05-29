from snfs_traffic.core import empty_state
from snfs_traffic.rl.episode import EpisodeConfig, compute_episode_flags


def _controlled_state():
    state = empty_state(1)
    state.controlled[0] = True
    return state


def test_truncated_false_before_max_steps():
    terminated, truncated = compute_episode_flags(
        _controlled_state(),
        controlled_vehicle_id=0,
        step_index=4,
        config=EpisodeConfig(max_steps=5),
    )
    assert terminated is False
    assert truncated is False


def test_truncated_true_at_max_steps():
    terminated, truncated = compute_episode_flags(
        _controlled_state(),
        controlled_vehicle_id=0,
        step_index=5,
        config=EpisodeConfig(max_steps=5),
    )
    assert terminated is False
    assert truncated is True


def test_terminated_false_for_normal_fixed_horizon_ring_road_operation():
    terminated, truncated = compute_episode_flags(
        _controlled_state(),
        controlled_vehicle_id=0,
        step_index=1,
        config=EpisodeConfig(max_steps=5),
    )
    assert terminated is False
    assert truncated is False


def test_terminated_true_if_controlled_vehicle_not_alive_when_enabled():
    state = _controlled_state()
    state.alive[0] = False

    terminated, truncated = compute_episode_flags(
        state,
        controlled_vehicle_id=0,
        step_index=1,
        config=EpisodeConfig(max_steps=5, terminate_on_no_alive_controlled=True),
    )
    assert terminated is True
    assert truncated is False


def test_missing_controlled_does_not_terminate_when_disabled():
    state = _controlled_state()
    state.controlled[0] = False

    terminated, truncated = compute_episode_flags(
        state,
        controlled_vehicle_id=0,
        step_index=1,
        config=EpisodeConfig(max_steps=5, terminate_on_no_alive_controlled=False),
    )
    assert terminated is False
    assert truncated is False
