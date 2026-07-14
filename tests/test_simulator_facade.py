import numpy as np

from snfs_traffic.control import LANE_LEFT, LateralOverrideResult
from snfs_traffic.core import SimulationParams, empty_state
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


def test_simulator_lateral_override_facade_records_rule_result():
    params = SimulationParams(num_lanes=2, road_length=40, p_lane_change=0.0)
    state = empty_state(1)
    state.lane[0] = 1
    state.pos[0] = 10
    sim = TrafficSimulator(params=params, backend="reference", validate=True)
    sim.reset(state=state, rng_seed=12)

    out = sim.step_lateral_overrides({int(state.vehicle_id[0]): LANE_LEFT})

    assert int(out.lane[0]) == 0
    assert sim.last_step_info.actions_supplied is True
    assert isinstance(sim.last_step_info.action_result, LateralOverrideResult)
    assert sim.last_step_info.backend_name == "reference+reference-override"
