from __future__ import annotations

from snfs_traffic.backends.optimized import OptimizedBackend, get_optimized_backend
from snfs_traffic.core.backend import ReferenceBackend, StepBackend, get_reference_backend
from snfs_traffic.core.indexing_numba import NUMBA_AVAILABLE as INDEX_NUMBA_AVAILABLE
from snfs_traffic.core.lane_change_numba import NUMBA_AVAILABLE as LANE_NUMBA_AVAILABLE
from snfs_traffic.core.longitudinal_numba import NUMBA_AVAILABLE as LONG_NUMBA_AVAILABLE

BACKEND_REFERENCE = "reference"
BACKEND_OPTIMIZED = "optimized"
BACKEND_AUTO = "auto"


def optimized_backend_available() -> bool:
    return bool(INDEX_NUMBA_AVAILABLE and LANE_NUMBA_AVAILABLE and LONG_NUMBA_AVAILABLE)


def get_backend(name: str = BACKEND_AUTO) -> StepBackend:
    normalized = name.strip().lower()
    if normalized == BACKEND_REFERENCE:
        return get_reference_backend()
    if normalized in {BACKEND_OPTIMIZED, BACKEND_AUTO}:
        if optimized_backend_available():
            return get_optimized_backend()
        return get_reference_backend()
    raise ValueError("unknown backend name. expected one of: 'auto', 'optimized', 'reference'")


__all__ = [
    "BACKEND_AUTO",
    "BACKEND_OPTIMIZED",
    "BACKEND_REFERENCE",
    "OptimizedBackend",
    "ReferenceBackend",
    "StepBackend",
    "get_backend",
    "get_optimized_backend",
    "get_reference_backend",
    "optimized_backend_available",
]
