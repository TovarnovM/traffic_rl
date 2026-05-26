import importlib
import sys


def test_visualization_imports_without_pygame_imageio_init():
    import snfs_traffic

    assert snfs_traffic.__version__
    assert "pygame" not in sys.modules
    assert "imageio" not in sys.modules

    from snfs_traffic.visualization import RoadRenderConfig, RoadRenderer, VideoWriter

    assert RoadRenderConfig is not None
    assert RoadRenderer is not None
    assert VideoWriter is not None
    assert "pygame" not in sys.modules
    assert "imageio" not in sys.modules
    importlib.invalidate_caches()
