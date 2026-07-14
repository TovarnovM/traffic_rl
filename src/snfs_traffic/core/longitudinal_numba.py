from __future__ import annotations
import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


def draw_longitudinal_randoms(alive: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw Revised S-NFS longitudinal random variates in reference order."""
    u_s = np.zeros(alive.shape[0], dtype=np.float64)
    u_q = np.zeros(alive.shape[0], dtype=np.float64)
    u_b = np.zeros(alive.shape[0], dtype=np.float64)
    for i in range(alive.shape[0]):
        if alive[i]:
            u_s[i] = float(rng.random())
            u_q[i] = float(rng.random())
            u_b[i] = float(rng.random())
    return u_s, u_q, u_b


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _advance_positions_numba(pos, vel, alive, road_length):
        out = pos.copy()
        for i in range(pos.shape[0]):
            if alive[i]:
                out[i] = (int(pos[i]) + int(vel[i])) % road_length
        return out


    @njit(cache=True)
    def _cumulative_forward_gap_unit_length_numba(i, k, pos_like, lane, lane_order, lane_counts, lane_rank, road_length):
        if k <= 0:
            return 0
        total = 0
        cur = int(i)
        for _ in range(int(k)):
            lane_cur = int(lane[cur])
            rank_cur = int(lane_rank[cur])
            count = int(lane_counts[lane_cur])

            if count <= 1:
                total += int(road_length) - 1
                continue

            front = int(lane_order[lane_cur, (rank_cur + 1) % count])
            total += int((int(pos_like[front]) - int(pos_like[cur]) - 1) % road_length)
            cur = front
        if total < 0:
            return 0
        return int(total)


    @njit(cache=True)
    def _compute_longitudinal_velocities_unit_length_numba(
        lane,
        pos,
        vel,
        alive,
        controlled,
        lane_order,
        lane_counts,
        lane_rank,
        u_s,
        u_q,
        u_b,
        road_length,
        vmax_default,
        vmax_controlled,
        G,
        q,
        r,
        S,
        P1,
        P2,
        P3,
        P4,
    ):
        n = int(vel.shape[0])
        out = vel.copy()
        v0 = np.empty((n,), dtype=np.int64)
        prev_pos = np.empty((n,), dtype=np.int64)
        v4 = np.empty((n,), dtype=np.int64)

        for i in range(n):
            v0[i] = int(vel[i])
            prev_pos[i] = (int(pos[i]) - int(v0[i])) % int(road_length)
            v4[i] = int(v0[i])

        for i in range(n):
            if not alive[i]:
                continue

            lane_i = int(lane[i])
            rank_i = int(lane_rank[i])
            count = int(lane_counts[lane_i])
            vmax_i = int(vmax_controlled if controlled[i] else vmax_default)

            if count <= 1:
                g = int(road_length) - 1
                leader_v = int(v0[i])
            else:
                leader = int(lane_order[lane_i, (rank_i + 1) % count])
                g = int((int(pos[leader]) - int(pos[i]) - 1) % road_length)
                leader_v = int(v0[leader])

            s_i = int(S if float(u_s[i]) < float(r) else 1)

            if g >= int(G) or int(v0[i]) <= leader_v:
                v1 = min(vmax_i, int(v0[i]) + 1)
            else:
                v1 = int(v0[i])

            if float(u_q[i]) < float(q):
                prev_gap = _cumulative_forward_gap_unit_length_numba(
                    i, s_i, prev_pos, lane, lane_order, lane_counts, lane_rank, road_length
                )
                v2 = min(v1, int(prev_gap))
            else:
                v2 = v1

            cur_gap = _cumulative_forward_gap_unit_length_numba(
                i, s_i, pos, lane, lane_order, lane_counts, lane_rank, road_length
            )
            v3 = min(v2, int(cur_gap))

            if g >= int(G):
                p_i = float(P1)
            elif int(v0[i]) < leader_v:
                p_i = float(P2)
            elif int(v0[i]) == leader_v:
                p_i = float(P3)
            else:
                p_i = float(P4)

            if float(u_b[i]) < (1.0 - p_i):
                if v3 > 0:
                    v4_i = max(v3 - 1, 1)
                else:
                    v4_i = 0
            else:
                v4_i = v3
            v4[i] = max(0, min(vmax_i, int(v4_i)))

        v_safe = v4.copy()
        for l in range(lane_order.shape[0]):
            count = int(lane_counts[l])
            if count <= 1:
                continue
            for _ in range(count):
                for rr in range(count - 1, -1, -1):
                    i = int(lane_order[l, rr])
                    j = int(lane_order[l, (rr + 1) % count])
                    if not alive[i]:
                        continue
                    g = int((int(pos[j]) - int(pos[i]) - 1) % road_length)
                    safe = g + int(v_safe[j])
                    if safe < int(v_safe[i]):
                        v_safe[i] = safe

        for i in range(n):
            if alive[i]:
                value = int(v_safe[i])
                if value < 0:
                    value = 0
                out[i] = value
        return out


    @njit(cache=True)
    def _compute_longitudinal_velocities_unit_length_controlled_speed_numba(
        lane,
        pos,
        vel,
        alive,
        controlled,
        has_speed_delta,
        speed_delta,
        lane_order,
        lane_counts,
        lane_rank,
        u_s,
        u_q,
        u_b,
        road_length,
        vmax_default,
        vmax_controlled,
        G,
        q,
        r,
        S,
        P1,
        P2,
        P3,
        P4,
    ):
        n = int(vel.shape[0])
        out = vel.copy()
        v0 = np.empty((n,), dtype=np.int64)
        prev_pos = np.empty((n,), dtype=np.int64)
        v4 = np.empty((n,), dtype=np.int64)

        for i in range(n):
            v0[i] = int(vel[i])
            prev_pos[i] = (int(pos[i]) - int(v0[i])) % int(road_length)
            v4[i] = int(v0[i])

        for i in range(n):
            if not alive[i]:
                continue

            lane_i = int(lane[i])
            rank_i = int(lane_rank[i])
            count = int(lane_counts[lane_i])
            vmax_i = int(vmax_controlled if controlled[i] else vmax_default)

            if count <= 1:
                g = int(road_length) - 1
                leader_v = int(v0[i])
            else:
                leader = int(lane_order[lane_i, (rank_i + 1) % count])
                g = int((int(pos[leader]) - int(pos[i]) - 1) % road_length)
                leader_v = int(v0[leader])

            if controlled[i] and has_speed_delta[i]:
                requested = int(v0[i]) + int(speed_delta[i])
                if requested < 0:
                    requested = 0
                if requested > int(vmax_controlled):
                    requested = int(vmax_controlled)
                v4[i] = requested
                continue

            s_i = int(S if float(u_s[i]) < float(r) else 1)

            if g >= int(G) or int(v0[i]) <= leader_v:
                v1 = min(vmax_i, int(v0[i]) + 1)
            else:
                v1 = int(v0[i])

            if float(u_q[i]) < float(q):
                prev_gap = _cumulative_forward_gap_unit_length_numba(
                    i, s_i, prev_pos, lane, lane_order, lane_counts, lane_rank, road_length
                )
                v2 = min(v1, int(prev_gap))
            else:
                v2 = v1

            cur_gap = _cumulative_forward_gap_unit_length_numba(
                i, s_i, pos, lane, lane_order, lane_counts, lane_rank, road_length
            )
            v3 = min(v2, int(cur_gap))

            if g >= int(G):
                p_i = float(P1)
            elif int(v0[i]) < leader_v:
                p_i = float(P2)
            elif int(v0[i]) == leader_v:
                p_i = float(P3)
            else:
                p_i = float(P4)

            if float(u_b[i]) < (1.0 - p_i):
                if v3 > 0:
                    v4_i = max(v3 - 1, 1)
                else:
                    v4_i = 0
            else:
                v4_i = v3
            v4[i] = max(0, min(vmax_i, int(v4_i)))

        v_safe = v4.copy()
        for l in range(lane_order.shape[0]):
            count = int(lane_counts[l])
            if count <= 1:
                continue
            for _ in range(count):
                for rr in range(count - 1, -1, -1):
                    i = int(lane_order[l, rr])
                    j = int(lane_order[l, (rr + 1) % count])
                    if not alive[i]:
                        continue
                    g = int((int(pos[j]) - int(pos[i]) - 1) % road_length)
                    safe = g + int(v_safe[j])
                    if safe < int(v_safe[i]):
                        v_safe[i] = safe

        for i in range(n):
            if alive[i]:
                value = int(v_safe[i])
                if value < 0:
                    value = 0
                out[i] = value
        return out, v4
else:
    def _advance_positions_numba(*args, **kwargs):
        raise ImportError("Numba is not installed")

    def _compute_longitudinal_velocities_unit_length_numba(*args, **kwargs):
        raise ImportError("Numba is not installed")

    def _compute_longitudinal_velocities_unit_length_controlled_speed_numba(*args, **kwargs):
        raise ImportError("Numba is not installed")


def advance_positions_numba(pos: np.ndarray, vel: np.ndarray, alive: np.ndarray, *, road_length: int) -> np.ndarray:
    if not NUMBA_AVAILABLE:
        raise ImportError("Numba is not installed")
    return _advance_positions_numba(pos, vel, alive, road_length)


def compute_longitudinal_velocities_numba(
    lane: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    alive: np.ndarray,
    controlled: np.ndarray,
    lane_order: np.ndarray,
    lane_counts: np.ndarray,
    lane_rank: np.ndarray,
    u_s: np.ndarray,
    u_q: np.ndarray,
    u_b: np.ndarray,
    *,
    road_length: int,
    vmax_default: int,
    vmax_controlled: int,
    G: int,
    q: float,
    r: float,
    S: int,
    P1: float,
    P2: float,
    P3: float,
    P4: float,
) -> np.ndarray:
    """Compute full Revised S-NFS velocities for unit-length vehicles with Numba."""
    if not NUMBA_AVAILABLE:
        raise ImportError("Numba is not installed")
    return _compute_longitudinal_velocities_unit_length_numba(
        lane,
        pos,
        vel,
        alive,
        controlled,
        lane_order,
        lane_counts,
        lane_rank,
        u_s,
        u_q,
        u_b,
        road_length,
        vmax_default,
        vmax_controlled,
        G,
        q,
        r,
        S,
        P1,
        P2,
        P3,
        P4,
    )


def compute_longitudinal_velocities_controlled_speed_numba(
    lane: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    alive: np.ndarray,
    controlled: np.ndarray,
    has_speed_delta: np.ndarray,
    speed_delta: np.ndarray,
    lane_order: np.ndarray,
    lane_counts: np.ndarray,
    lane_rank: np.ndarray,
    u_s: np.ndarray,
    u_q: np.ndarray,
    u_b: np.ndarray,
    *,
    road_length: int,
    vmax_default: int,
    vmax_controlled: int,
    G: int,
    q: float,
    r: float,
    S: int,
    P1: float,
    P2: float,
    P3: float,
    P4: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unit-length velocities with explicit controlled speed deltas.

    Returns final safety-clipped velocities and desired pre-safety velocities.
    """
    if not NUMBA_AVAILABLE:
        raise ImportError("Numba is not installed")
    return _compute_longitudinal_velocities_unit_length_controlled_speed_numba(
        lane,
        pos,
        vel,
        alive,
        controlled,
        has_speed_delta,
        speed_delta,
        lane_order,
        lane_counts,
        lane_rank,
        u_s,
        u_q,
        u_b,
        road_length,
        vmax_default,
        vmax_controlled,
        G,
        q,
        r,
        S,
        P1,
        P2,
        P3,
        P4,
    )
