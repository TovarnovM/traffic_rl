"""Backend contract and reference backend implementation.

A step backend performs exactly one full simulator step and must match
:func:`snfs_traffic.core.step_reference` semantics.

Contract requirements:
- Inputs: ``TrafficState``, ``SimulationParams``, ``RingTopology``, and
  ``np.random.Generator``.
- Output: next ``TrafficState`` after one full step.
- Randomness must be consumed only from the provided ``rng`` argument.
- For the same initial state and RNG seed, backend output must match
  ``step_reference``.
- Occupancy/collision/gap semantics remain head-cell-only.
- ``changed_lane`` / ``last_lane_delta`` semantics must describe lateral
  movement during the returned full step.
- Controlled RL action semantics are not implemented.
- Length-aware occupancy and body-cell geometry are not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from snfs_traffic.core.params import SimulationParams
from snfs_traffic.core.state import TrafficState
from snfs_traffic.core.step_reference import step_reference
from snfs_traffic.topology import RingTopology


class StepBackend(Protocol):
    """Protocol for one full simulator step backend."""

    name: str

    def step(
        self,
        state: TrafficState,
        params: SimulationParams,
        topology: RingTopology,
        rng: np.random.Generator,
    ) -> TrafficState:
        """Run one full step and return the next state."""

        ...


@dataclass(frozen=True)
class ReferenceBackend:
    """Backend wrapper around the reference implementation."""

    name: str = "reference"

    def step(
        self,
        state: TrafficState,
        params: SimulationParams,
        topology: RingTopology,
        rng: np.random.Generator,
    ) -> TrafficState:
        return step_reference(state, params, topology, rng)


_REFERENCE_BACKEND = ReferenceBackend()


def get_reference_backend() -> ReferenceBackend:
    """Return the singleton reference backend."""

    return _REFERENCE_BACKEND
