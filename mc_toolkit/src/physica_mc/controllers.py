"""Control mechanics for one episode: common K=1 and H_C private deadlines.

The three concerns are deliberately kept apart:

* the common AV-action mask is built by ``build_pv_graph_observation`` and its
  K=1 cooldown is tracked by :class:`CommonCooldown`;
* H_C adds a *private* deadline per physical vehicle on top of that common
  interface without changing the common mask or its denominator;
* the fixed actions themselves (:func:`stay_action`, :func:`left_if_safe_action`)
  are the unchanged E2 semantics.

No RNG is drawn here.  ``LateralOverrideResult`` is consumed after the physics
step; a rejected/stay/native lane change never starts an H_C timer.
"""

from __future__ import annotations

import numpy as np

from snfs_traffic.control import LateralOverrideResult


# Same action space as the historical graph interface: 0=left, 1=stay, 2=right.
ACTION_INDEX_TO_DELTA = np.asarray([-1, 0, 1], dtype=np.int8)
H_COOLDOWNS = (1, 2, 4, 8)


def _nonnegative_clock(value, name="pre-step time"):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _graph_masks(graph):
    node_mask = np.asarray(graph.node_mask)
    action_mask = np.asarray(graph.action_mask)
    vehicle_id = np.asarray(graph.vehicle_id)
    if node_mask.ndim != 1 or action_mask.shape != (node_mask.shape[0], 3):
        raise ValueError("graph node/action masks have incompatible shapes")
    if vehicle_id.shape != node_mask.shape:
        raise ValueError("graph vehicle_id does not match the node axis")
    return node_mask, action_mask, vehicle_id


def stay_action(graph) -> np.ndarray:
    """Explicit stay for every padded/active slot (graph noop semantics)."""

    node_mask, _action_mask, _vehicle_id = _graph_masks(graph)
    return np.ones(node_mask.shape, dtype=np.int64)


def left_if_safe_action(graph) -> np.ndarray:
    """The unchanged original left-if-safe rule in graph action indices."""

    node_mask, action_mask, _vehicle_id = _graph_masks(graph)
    action = np.ones(node_mask.shape, dtype=np.int64)
    active = node_mask.astype(bool)
    action[active & action_mask[:, 0].astype(bool)] = 0
    return action


def common_directions(graph) -> tuple[np.ndarray, np.ndarray]:
    """Common-mask legal left/right opportunities, private H_C excluded."""

    node_mask, action_mask, _vehicle_id = _graph_masks(graph)
    active = node_mask.astype(bool)
    return active & action_mask[:, 0].astype(bool), active & action_mask[:, 2].astype(bool)


class CommonCooldown:
    """The unchanged environment/common K=1 cooldown keyed by physical ID.

    ``K`` is fixed to one for protocol v1.  An applied nonstay command installs
    one remaining suppressed step; ``decide``-time ticking is handled by the
    graph observation (which zeroes nonstay mask columns).  The explicit
    suppression pass exists to keep the request/application accounting honest
    if a controller ever emits outside the mask.
    """

    K = 1

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._remaining: dict[int, int] = {}

    def remaining(self) -> dict[int, int]:
        return dict(self._remaining)

    def update(self, result: LateralOverrideResult | None) -> None:
        next_remaining = {
            vehicle_id: remaining - 1
            for vehicle_id, remaining in self._remaining.items()
            if remaining > 1
        }
        if result is not None:
            for vehicle_id, applied in zip(result.vehicle_id, result.applied):
                if bool(applied):
                    next_remaining[int(vehicle_id)] = 1
        self._remaining = next_remaining

    def suppress(self, requests: dict[int, int]) -> tuple[dict[int, int], int]:
        """Zero nonstay requests under common K=1; return (commands, count)."""

        commands: dict[int, int] = {}
        suppressed = 0
        for vehicle_id, delta in requests.items():
            vehicle_id, delta = int(vehicle_id), int(delta)
            if delta != 0 and self._remaining.get(vehicle_id, 0) > 0:
                commands[vehicle_id] = 0
                suppressed += 1
            else:
                commands[vehicle_id] = delta
        return commands, suppressed


class OwnIntensityController:
    """H_C over left-if-safe with pre-step clocks and physical-ID deadlines.

    Call :meth:`action` once, apply the returned action, then call
    :meth:`observe` with the private full result (or ``None`` for a native
    step).  A successful own nonstay request emitted at pre-step ``t`` sets
    ``next_allowed = t + C + 1``; decisions at ``t+1 ... t+C`` are explicit
    stays that keep the vehicle's node active.  ``reset`` is required for every
    new trajectory, including physical-ID reuse.
    """

    def __init__(self, C: int) -> None:
        if (
            isinstance(C, (bool, np.bool_))
            or not isinstance(C, (int, np.integer))
            or int(C) not in H_COOLDOWNS
        ):
            raise ValueError("C must be an integer in {1, 2, 4, 8}")
        self.C = int(C)
        self.reset()

    def reset(self) -> None:
        self.next_allowed: dict[int, int] = {}
        self._last_time = -1
        self._pending: dict[int, int] | None = None
        self._pending_time: int | None = None
        self.last_throttled_decisions = 0
        self.last_suppressed_left_opportunities = 0

    def action(self, graph, pretime: int) -> np.ndarray:
        t = _nonnegative_clock(pretime)
        if self._pending is not None:
            raise RuntimeError("observe the pending action before another action")
        if t <= self._last_time:
            raise ValueError("pretime must advance; reset on a new trajectory")
        node_mask, action_mask, vehicle_id = _graph_masks(graph)
        action = left_if_safe_action(graph)
        throttled = 0
        suppressed_left = 0
        pending: dict[int, int] = {}
        for slot in np.flatnonzero(node_mask):
            vid = int(vehicle_id[slot])
            if t < self.next_allowed.get(vid, 0):
                throttled += 1
                if bool(action_mask[slot, 0]):
                    suppressed_left += 1
                # Explicit stay, not omission: the ID must stay active.
                action[slot] = 1
            pending[vid] = int(ACTION_INDEX_TO_DELTA[int(action[slot])])
        self._pending = pending
        self._pending_time = t
        self.last_throttled_decisions = throttled
        self.last_suppressed_left_opportunities = suppressed_left
        return action

    def observe(self, result: LateralOverrideResult | None, pretime: int) -> None:
        t = _nonnegative_clock(pretime)
        if self._pending is None or t != self._pending_time:
            raise ValueError("feedback must match the pending pre-action time")
        if result is not None:
            if not isinstance(result, LateralOverrideResult):
                raise TypeError("private full LateralOverrideResult required")
            for vehicle_id, requested, applied in zip(
                result.vehicle_id, result.requested_lane_delta, result.applied
            ):
                vehicle_id, delta = int(vehicle_id), int(requested)
                if (
                    bool(applied)
                    and delta != 0
                    and self._pending.get(vehicle_id) == delta
                ):
                    self.next_allowed[vehicle_id] = t + self.C + 1
        self._last_time = t
        self._pending = None
        self._pending_time = None
