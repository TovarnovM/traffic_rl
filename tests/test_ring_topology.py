import numpy as np
import pytest

from snfs_traffic.topology import RingTopology


def test_ring_topology_valid_construction() -> None:
    topology = RingTopology(num_lanes=4, length=1500)
    assert topology.num_lanes == 4
    assert topology.length == 1500
    assert topology.boundary == "periodic"


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"num_lanes": 0, "length": 1500}, "num_lanes"),
        ({"num_lanes": -1, "length": 1500}, "num_lanes"),
        ({"num_lanes": True, "length": 1500}, "num_lanes"),
        ({"num_lanes": 4.0, "length": 1500}, "num_lanes"),
        ({"num_lanes": 4, "length": 0}, "length"),
        ({"num_lanes": 4, "length": -10}, "length"),
        ({"num_lanes": 4, "length": True}, "length"),
        ({"num_lanes": 4, "length": 1500.0}, "length"),
    ],
)
def test_ring_topology_invalid_construction(kwargs: dict, field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        RingTopology(**kwargs)


def test_normalize_pos_scalar_behavior() -> None:
    topology = RingTopology(num_lanes=1, length=10)
    assert topology.normalize_pos(0) == 0
    assert topology.normalize_pos(9) == 9
    assert topology.normalize_pos(10) == 0
    assert topology.normalize_pos(12) == 2
    assert topology.normalize_pos(-1) == 9
    assert topology.normalize_pos(-11) == 9


def test_normalize_pos_ndarray_behavior() -> None:
    topology = RingTopology(num_lanes=1, length=10)
    pos = np.array([-11, -1, 0, 9, 10, 12], dtype=np.int32)
    pos_before = pos.copy()

    out = topology.normalize_pos(pos)

    np.testing.assert_array_equal(out, np.array([9, 9, 0, 9, 0, 2]))
    np.testing.assert_array_equal(pos, pos_before)


def test_forward_distance_scalar_behavior() -> None:
    topology = RingTopology(num_lanes=1, length=10)
    assert topology.forward_distance(0, 0) == 0
    assert topology.forward_distance(0, 1) == 1
    assert topology.forward_distance(1, 0) == 9
    assert topology.forward_distance(8, 1) == 3
    assert topology.forward_distance(9, 0) == 1
    assert topology.forward_distance(2, 7) == 5
    assert topology.forward_distance(11, 13) == 2
    assert topology.forward_distance(-1, 1) == 2


def test_forward_distance_ndarray_behavior() -> None:
    topology = RingTopology(num_lanes=1, length=10)
    from_pos = np.array([0, 1, 8, 9], dtype=np.int32)
    to_pos = np.array([0, 0, 1, 0], dtype=np.int32)

    out = topology.forward_distance(from_pos, to_pos)
    np.testing.assert_array_equal(out, np.array([0, 9, 3, 1]))

    broadcast_out = topology.forward_distance(9, np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(broadcast_out, np.array([1, 2, 3]))


def test_signed_delta_scalar_behavior() -> None:
    topology = RingTopology(num_lanes=1, length=10)
    assert topology.signed_delta(0, 0) == 0
    assert topology.signed_delta(0, 1) == 1
    assert topology.signed_delta(1, 0) == -1
    assert topology.signed_delta(9, 1) == 2
    assert topology.signed_delta(1, 9) == -2
    assert topology.signed_delta(0, 5) == 5
    assert topology.signed_delta(0, 6) == -4


def test_signed_delta_odd_length_behavior() -> None:
    topology = RingTopology(num_lanes=1, length=9)
    assert topology.signed_delta(0, 4) == 4
    assert topology.signed_delta(0, 5) == -4


def test_signed_delta_ndarray_behavior() -> None:
    topology = RingTopology(num_lanes=1, length=10)
    from_pos = np.array([0, 1, 9, 0], dtype=np.int32)
    to_pos = np.array([1, 0, 1, 6], dtype=np.int32)

    out = topology.signed_delta(from_pos, to_pos)
    np.testing.assert_array_equal(out, np.array([1, -1, 2, -4]))


def test_invalid_position_inputs() -> None:
    topology = RingTopology(num_lanes=1, length=10)

    with pytest.raises(ValueError):
        topology.normalize_pos(True)

    with pytest.raises(ValueError):
        topology.normalize_pos(1.5)

    with pytest.raises(ValueError):
        topology.normalize_pos(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ValueError):
        topology.forward_distance(0, 1.5)

    with pytest.raises(ValueError):
        topology.signed_delta(np.array([True, False], dtype=np.bool_), 0)


def test_public_import() -> None:
    from snfs_traffic.topology import RingTopology as ImportedRingTopology

    topology = ImportedRingTopology(num_lanes=2, length=20)
    assert isinstance(topology, RingTopology)
