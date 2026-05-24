from __future__ import annotations
import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _advance_positions_numba(pos, vel, alive, road_length):
        out = pos.copy()
        for i in range(pos.shape[0]):
            if alive[i]:
                out[i] = (int(pos[i]) + int(vel[i])) % road_length
        return out
else:
    def _advance_positions_numba(*args, **kwargs):
        raise ImportError("Numba is not installed")


def advance_positions_numba(pos: np.ndarray, vel: np.ndarray, alive: np.ndarray, *, road_length: int) -> np.ndarray:
    if not NUMBA_AVAILABLE:
        raise ImportError("Numba is not installed")
    return _advance_positions_numba(pos, vel, alive, road_length)
