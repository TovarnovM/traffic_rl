import numpy as np

from snfs_traffic.core import SimulationParams
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.simulator import TrafficSimulator


def _assert_state_eq(a, b):
    for f in a.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))


def test_reset_state_copy_and_mutation_isolated():
    p = SimulationParams(num_lanes=3, road_length=40)
    init = make_uniform_random_state(num_lanes=3, road_length=40, density=0.2, seed=3)
    sim = TrafficSimulator(params=p)
    out = sim.reset(state=init)
    out.lane[:] = 0
    np.testing.assert_array_equal(sim.state.lane, init.lane)


def test_validate_and_backend_selection_and_parity():
    params = SimulationParams(num_lanes=3, road_length=40)
    init = make_uniform_random_state(num_lanes=3, road_length=40, density=0.2, seed=3, vehicle_mix=VehicleMix(controlled_fraction=0.2))
    sim_ref = TrafficSimulator(params=params, backend="reference", validate=True)
    sim_auto = TrafficSimulator(params=params, backend="auto", validate=True)
    sim_ref.reset(state=init, rng_seed=9)
    sim_auto.reset(state=init, rng_seed=9)
    backend = sim_ref.backend
    direct = init.copy(); rng = np.random.default_rng(9)
    for _ in range(5):
        direct = backend.step(direct, params, sim_ref.topology, rng)
        got = sim_ref.step(None)
        _assert_state_eq(direct, got)
    sim_auto.step(None)
