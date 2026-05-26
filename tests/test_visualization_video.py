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


def test_video_writer_frame_validation(tmp_path):
    p = tmp_path / "out.mp4"
    writer = VideoWriter(p, fps=12)

    with pytest.raises(ValueError):
        writer.write(np.zeros((16, 16), dtype=np.uint8))

    with pytest.raises(ValueError):
        writer.write(np.zeros((16, 16, 3), dtype=np.float32))

    with pytest.raises(ValueError):
        writer.write([[0, 0, 0]])

    assert writer._writer is None
