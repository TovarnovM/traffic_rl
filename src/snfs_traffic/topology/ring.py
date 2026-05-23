from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import PositionInput, _validate_integer_positions, _validate_positive_int


@dataclass(frozen=True, slots=True)
class RingTopology:
    num_lanes: int
    length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_lanes", _validate_positive_int("num_lanes", self.num_lanes))
        object.__setattr__(self, "length", _validate_positive_int("length", self.length))

    @property
    def boundary(self) -> str:
        return "periodic"

    def normalize_pos(self, pos: PositionInput) -> int | np.ndarray:
        pos_checked = _validate_integer_positions("pos", pos)
        normalized = np.mod(pos_checked, self.length)
        if isinstance(normalized, np.ndarray):
            return normalized
        return int(normalized)

    def forward_distance(self, from_pos: PositionInput, to_pos: PositionInput) -> int | np.ndarray:
        from_checked = _validate_integer_positions("from_pos", from_pos)
        to_checked = _validate_integer_positions("to_pos", to_pos)

        distance = np.mod(to_checked - from_checked, self.length)
        if isinstance(distance, np.ndarray):
            return distance
        return int(distance)

    def signed_delta(self, from_pos: PositionInput, to_pos: PositionInput) -> int | np.ndarray:
        d = self.forward_distance(from_pos, to_pos)
        threshold = self.length / 2

        if isinstance(d, np.ndarray):
            return np.where(d > threshold, d - self.length, d)

        if d > threshold:
            return d - self.length
        return d
