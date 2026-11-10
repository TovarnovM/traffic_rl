"""Hand-built control fixtures; no simulator, observer, RNG or torch imports."""

from types import SimpleNamespace

import numpy as np
import pytest

from physica_mc.controllers import (
    ACTION_INDEX_TO_DELTA,
    CommonCooldown,
    OwnIntensityController,
    common_directions,
    left_if_safe_action,
    stay_action,
)
from snfs_traffic.control import LateralOverrideResult


def graph(ids, left=None, right=None):
    ids = tuple(int(value) for value in ids)
    left = np.zeros(len(ids), dtype=np.int8) if left is None else np.asarray(left, dtype=np.int8)
    right = np.zeros(len(ids), dtype=np.int8) if right is None else np.asarray(right, dtype=np.int8)
    node_mask = np.zeros(len(ids), dtype=np.int8)
    action_mask = np.zeros((len(ids), 3), dtype=np.int8)
    action_mask[:, 1] = 1
    for slot, vehicle_id in enumerate(ids):
        if vehicle_id < 0:
            continue
        node_mask[slot] = 1
        action_mask[slot, 0] = int(left[slot])
        action_mask[slot, 2] = int(right[slot])
    return SimpleNamespace(
        node_mask=node_mask,
        action_mask=action_mask,
        vehicle_id=np.asarray(ids, dtype=np.int32),
    )


def outcome(ids, requested, applied, reasons=None):
    ids = np.asarray(ids, dtype=np.int32)
    requested = np.asarray(requested, dtype=np.int8)
    applied = np.asarray(applied, dtype=bool)
    n = ids.size
    return LateralOverrideResult(
        vehicle_id=ids,
        requested_lane_delta=requested,
        applied_lane_delta=np.where(applied, requested, 0).astype(np.int8),
        valid=applied.copy(),
        applied=applied,
        rejection_reason=tuple(reasons if reasons is not None else [""] * n),
    )


def test_stay_and_left_if_safe_use_only_the_common_mask():
    g = graph([7, 8, -1], left=[1, 0, 0])
    np.testing.assert_array_equal(stay_action(g), [1, 1, 1])
    np.testing.assert_array_equal(left_if_safe_action(g), [0, 1, 1])
    left, right = common_directions(g)
    np.testing.assert_array_equal(left, [True, False, False])
    np.testing.assert_array_equal(right, [False, False, False])


def test_action_delta_contract():
    np.testing.assert_array_equal(ACTION_INDEX_TO_DELTA, [-1, 0, 1])


def test_own_intensity_private_timeline_updates_only_on_applied_own_request():
    C = 2
    controller = OwnIntensityController(C)
    g = graph([7], left=[1])

    for t in range(5):
        action = controller.action(g, t)
        if t in (0, 3):
            assert int(action[0]) == 0
            assert controller.last_throttled_decisions == 0
            assert controller.last_suppressed_left_opportunities == 0
        else:
            # Decisions t+1..t+C are explicit stays; the ID stays active.
            assert int(action[0]) == 1
            assert controller.last_throttled_decisions == 1
            assert controller.last_suppressed_left_opportunities == 1
        requested = -1 if t in (0, 3) else 0
        controller.observe(outcome([7], [requested], [True]), t)

    assert controller.next_allowed == {7: 6}


def test_own_intensity_rejected_or_stay_does_not_extend():
    controller = OwnIntensityController(2)
    g = graph([7], left=[1])

    assert int(controller.action(g, 0)[0]) == 0
    controller.observe(outcome([7], [-1], [False], ["unsafe"]), 0)
    assert controller.next_allowed == {}

    assert int(controller.action(g, 1)[0]) == 0
    controller.observe(outcome([7], [0], [True]), 1)
    assert controller.next_allowed == {}

    # An own action is a left request; a different applied request cannot start it.
    assert int(controller.action(g, 2)[0]) == 0
    controller.observe(outcome([7], [1], [True]), 2)
    assert controller.next_allowed == {}

    assert int(controller.action(g, 3)[0]) == 0
    controller.observe(outcome([7], [-1], [True]), 3)
    assert controller.next_allowed == {7: 3 + 2 + 1}


def test_own_intensity_none_feedback_and_reset():
    controller = OwnIntensityController(4)
    g = graph([7], left=[1])
    controller.action(g, 0)
    controller.observe(None, 0)
    assert controller.next_allowed == {}

    controller.action(g, 1)
    controller.observe(outcome([7], [-1], [True]), 1)
    assert controller.next_allowed == {7: 6}

    controller.reset()
    assert controller.next_allowed == {}
    assert controller.action(g, 0).tolist() == [0]
    controller.observe(None, 0)


def test_own_intensity_empty_active_set_does_not_stop_time():
    controller = OwnIntensityController(1)
    g = graph([-1, -1])
    action = controller.action(g, 0)
    np.testing.assert_array_equal(action, [1, 1])
    assert controller.last_throttled_decisions == 0
    assert controller.last_suppressed_left_opportunities == 0
    controller.observe(None, 0)
    assert controller.action(g, 1).tolist() == [1, 1]
    controller.observe(None, 1)


def test_own_intensity_deadlines_follow_physical_ids_not_slots():
    controller = OwnIntensityController(2)
    controller.next_allowed = {8: 5}
    g = graph([7, 8], left=[1, 1])
    action = controller.action(g, 0)
    np.testing.assert_array_equal(action, [0, 1])
    assert controller.last_suppressed_left_opportunities == 1
    controller.observe(outcome([7, 8], [-1, 0], [True, True]), 0)
    assert controller.next_allowed == {8: 5, 7: 3}


def test_own_intensity_requires_pending_protocol_and_advancing_clock():
    controller = OwnIntensityController(1)
    g = graph([7], left=[1])
    with pytest.raises(RuntimeError):
        controller.action(g, 0)
        controller.action(g, 0)
    controller.observe(None, 0)
    with pytest.raises(ValueError):
        controller.observe(None, 0)
    with pytest.raises(ValueError):
        controller.action(g, 0)  # pretime did not advance
    for bad in (-1, True, 1.0):
        with pytest.raises(ValueError):
            _ = OwnIntensityController(bad)


def test_own_intensity_rejects_wrong_feedback_type():
    controller = OwnIntensityController(1)
    controller.action(graph([7], left=[1]), 0)
    with pytest.raises(TypeError):
        controller.observe(object(), 0)


def test_malformed_graph_is_rejected():
    bad = SimpleNamespace(
        node_mask=np.zeros(2, dtype=np.int8),
        action_mask=np.zeros((2, 2), dtype=np.int8),
        vehicle_id=np.zeros(2, dtype=np.int32),
    )
    with pytest.raises(ValueError):
        left_if_safe_action(bad)


def test_common_cooldown_k1_suppression_and_reset():
    cooldown = CommonCooldown()
    commands, suppressed = cooldown.suppress({5: -1, 6: 0})
    assert commands == {5: -1, 6: 0} and suppressed == 0

    cooldown.update(outcome([5], [-1], [True]))
    assert cooldown.remaining() == {5: 1}
    commands, suppressed = cooldown.suppress({5: -1, 6: -1})
    assert commands == {5: 0, 6: -1} and suppressed == 1

    # After one suppressed decision the K=1 window has passed.
    cooldown.update(None)
    assert cooldown.remaining() == {}
    commands, suppressed = cooldown.suppress({5: -1})
    assert commands == {5: -1} and suppressed == 0

    cooldown.update(outcome([5], [-1], [False], ["unsafe"]))
    assert cooldown.remaining() == {}
    cooldown.reset()
    assert cooldown.remaining() == {}


def test_common_cooldown_does_not_count_explicit_stays():
    cooldown = CommonCooldown()
    cooldown.update(outcome([5, 6], [0, 0], [False, False]))
    assert cooldown.remaining() == {}
