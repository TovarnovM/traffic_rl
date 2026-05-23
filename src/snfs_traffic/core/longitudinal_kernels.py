"""Internal pure-array kernels for longitudinal same-lane phase."""

from __future__ import annotations

import numpy as np


def compute_longitudinal_velocities_kernel(
    vel: np.ndarray,
    alive: np.ndarray,
    controlled: np.ndarray,
    front_id: np.ndarray,
    front_gap: np.ndarray,
    *,
    road_length: int,
    vmax_default: int,
    vmax_controlled: int,
    G: int,
    S: int,
    r: float,
    P2: float,
    P3: float,
    P4: float,
    rng,
) -> np.ndarray:
    """Compute next-step velocities from validated longitudinal arrays.

    Inputs are treated as read-only; output preserves vel dtype.
    """

    out_i64 = vel.astype(np.int64, copy=True)

    for i in range(vel.shape[0]):
        if not alive[i]:
            continue

        old_v = int(vel[i])
        vmax_i = int(vmax_controlled if controlled[i] else vmax_default)
        fid = int(front_id[i])

        if fid == -1:
            gap_i = int(road_length - 1)
            leader_v = vmax_i
        else:
            gap_i = int(front_gap[i])
            leader_v = int(vel[fid])

        v = min(old_v + 1, vmax_i)

        if fid != -1 and gap_i <= G:
            anticipated_leader_motion = max(leader_v - int(S), 0)
            anticipated_gap = gap_i + anticipated_leader_motion
            v = min(v, anticipated_gap)

        if old_v == 0 and v > 0 and rng.random() < r:
            v = 0

        brake_prob = float(P4)
        if fid != -1 and gap_i <= S:
            brake_prob = max(brake_prob, 1.0 - float(P2))
        elif fid != -1 and gap_i <= G:
            brake_prob = max(brake_prob, 1.0 - float(P3))

        if v > 0 and rng.random() < brake_prob:
            v -= 1

        if fid != -1:
            v = min(v, gap_i)
        v = max(0, min(v, vmax_i))

        out_i64[i] = v

    return out_i64.astype(vel.dtype, copy=False)


def advance_positions_kernel(
    pos: np.ndarray,
    vel: np.ndarray,
    alive: np.ndarray,
    *,
    road_length: int,
) -> np.ndarray:
    """Advance periodic-ring positions with already computed velocities."""

    out = pos.copy()
    for i in range(pos.shape[0]):
        if not alive[i]:
            continue
        out[i] = (int(pos[i]) + int(vel[i])) % road_length
    return out
