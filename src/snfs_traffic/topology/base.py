from __future__ import annotations

from typing import Protocol

import numpy as np

PositionInput = int | np.integer | np.ndarray


def _validate_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _validate_integer_positions(name: str, value: PositionInput) -> int | np.integer | np.ndarray:
    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{name} must have integer dtype")
        return value

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer position")

    if isinstance(value, (int, np.integer)):
        return value

    raise ValueError(f"{name} must be an integer position")


class SupportsRingDistances(Protocol):
    num_lanes: int
    length: int

    @property
    def boundary(self) -> str:
        ...

    def normalize_pos(self, pos: PositionInput) -> int | np.ndarray:
        ...

    def forward_distance(self, from_pos: PositionInput, to_pos: PositionInput) -> int | np.ndarray:
        ...

    def signed_delta(self, from_pos: PositionInput, to_pos: PositionInput) -> int | np.ndarray:
        ...
