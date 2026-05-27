import numpy as np

from snfs_traffic.core import SimulationParams
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.simulator import ScenarioConfig, TrafficSimulator


def _assert_state_eq(a, b):
    for f in a.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))


def test_step_before_reset_raises():
    sim = TrafficSimulator(params=SimulationParams(num_lanes=3, road_length=30))
    try:
        sim.step()
        assert False
    except RuntimeError:
        assert True


def test_observe_before_reset_raises():
    sim = TrafficSimulator(params=SimulationParams(num_lanes=3, road_length=30))
    try:
        sim.observe()
        assert False
    except RuntimeError:
        assert True


def test_actions_none_parity_multistep():
    params = SimulationParams(num_lanes=3, road_length=40)
    init = make_uniform_random_state(num_lanes=3, road_length=40, density=0.2, seed=3, vehicle_mix=VehicleMix(controlled_fraction=0.2))
    sim = TrafficSimulator(params=params, backend="reference", validate=True)
    sim.reset(state=init, rng_seed=9)
    backend = sim.backend
    direct = init.copy(); rng = np.random.default_rng(9)
    for _ in range(5):
        direct = backend.step(direct, params, sim.topology, rng)
        got = sim.step(None)
        _assert_state_eq(direct, got)
