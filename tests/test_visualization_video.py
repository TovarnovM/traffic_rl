import numpy as np
import pytest

from snfs_traffic.visualization import VideoWriter


def test_video_writer_smoke(tmp_path):
    pytest.importorskip("imageio")
    p = tmp_path / "out.mp4"
    f = np.zeros((16, 16, 3), dtype=np.uint8)
    with VideoWriter(p, fps=12) as w:
        w.write(f)
        w.write_many([f])
    assert p.exists()
    assert p.stat().st_size > 0
