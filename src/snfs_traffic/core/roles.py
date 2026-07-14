"""Vehicle-role helpers shared by simulation, scenarios, and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .state import TrafficState


# Keep the existing scenario type ids stable (HDV=0, AV=1, BUS=2).
PRIORITY_VEH_TYPE = 3


def priority_vehicle_mask(
    state: "TrafficState", *, alive_only: bool = False
) -> np.ndarray:
    """Return a boolean mask selecting priority vehicles (PV)."""

    mask = np.asarray(state.veh_type == PRIORITY_VEH_TYPE, dtype=np.bool_)
    if alive_only:
        mask = mask & state.alive
    return np.ascontiguousarray(mask)


def high_speed_vehicle_mask(state: "TrafficState") -> np.ndarray:
    """Select vehicles that use ``vmax_controlled``.

    Priority vehicles retain native Revised S-NFS lane-changing behavior.  The
    mask is only the speed-limit selector passed into the legacy kernels.
    """

    return np.ascontiguousarray(state.controlled | priority_vehicle_mask(state))
