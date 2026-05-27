import numpy as np

from snfs_traffic.core import SimulationParams
from snfs_traffic.simulator import TrafficSimulator


def test_snapshot_is_copy_and_action_provider_gets_copy_and_step():
    sim = TrafficSimulator(params=SimulationParams(num_lanes=3, road_length=30))
    sim.reset()
    snap = sim.snapshot()
    snap.state.lane[:] = 0
    assert not np.all(sim.state.lane == 0)

    seen = {}

    def provider(state, step):
        state.lane[:] = 0
        seen["step"] = step
        return None

    list(sim.iter_rollout(steps=1, action_provider=provider))
    assert seen["step"] == 0


def test_rollout_equals_iter_from_same_state():
    p = SimulationParams(num_lanes=3, road_length=30)
    sim1 = TrafficSimulator(params=p)
    s0 = sim1.reset(seed=7)
    a = list(sim1.iter_rollout(steps=3, include_initial=True))

    sim2 = TrafficSimulator(params=p)
    sim2.reset(state=s0, rng_seed=7)
    b = sim2.rollout(steps=3, include_initial=True)
    assert [x.step for x in a] == [x.step for x in b]
