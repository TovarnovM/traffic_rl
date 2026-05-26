from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class VideoWriter:
    def __init__(self, path: str | Path, *, fps: int = 48) -> None:
        self.path = Path(path)
        self.fps = int(fps)
        self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def _require_imageio(self):
        try:
            import imageio.v2 as imageio
        except ImportError:
            try:
                import imageio
            except ImportError as exc:
                raise RuntimeError("VideoWriter requires visualization dependencies. Install with: python -m pip install -e '.[viz]'") from exc
        return imageio

    def _ensure_open(self) -> None:
        if self._writer is None:
            imageio = self._require_imageio()
            self._writer = imageio.get_writer(self.path, fps=self.fps)

    def write(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be np.uint8 RGB array with shape (H, W, 3)")
        self._ensure_open()
        self._writer.append_data(frame)

    def write_many(self, frames: Iterable[np.ndarray]) -> None:
        for frame in frames:
            self.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
