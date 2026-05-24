import pytest

import snfs_traffic.backends as backends
from snfs_traffic.backends import BACKEND_AUTO, BACKEND_OPTIMIZED, BACKEND_REFERENCE, get_backend, optimized_backend_available


def test_get_backend_reference_name():
    assert get_backend(BACKEND_REFERENCE).name == "reference"


def test_get_backend_auto_follows_availability():
    expected = "optimized" if optimized_backend_available() else "reference"
    assert get_backend(BACKEND_AUTO).name == expected


def test_get_backend_optimized_follows_documented_fallback():
    expected = "optimized" if optimized_backend_available() else "reference"
    assert get_backend(BACKEND_OPTIMIZED).name == expected


def test_get_backend_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="unknown backend name"):
        get_backend("bad_backend")


def test_get_backend_falls_back_when_one_kernel_unavailable(monkeypatch):
    monkeypatch.setattr(backends, "INDEX_NUMBA_AVAILABLE", False)
    monkeypatch.setattr(backends, "LANE_NUMBA_AVAILABLE", True)
    monkeypatch.setattr(backends, "LONG_NUMBA_AVAILABLE", True)
    assert optimized_backend_available() is False
    assert get_backend(BACKEND_AUTO).name == "reference"
    assert get_backend(BACKEND_OPTIMIZED).name == "reference"
