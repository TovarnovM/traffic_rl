from snfs_traffic.core import SimulationParams
from snfs_traffic.simulator import TrafficSimulator


def test_iter_rollout_include_initial():
    sim = TrafficSimulator(params=SimulationParams(num_lanes=3, road_length=30))
    sim.reset()
    snaps = list(sim.iter_rollout(steps=0, include_initial=True))
    assert len(snaps) == 1
    assert snaps[0].step == 0


def test_rollout_equiv_iter():
    sim = TrafficSimulator(params=SimulationParams(num_lanes=3, road_length=30))
    sim.reset()
    a = list(sim.iter_rollout(steps=3))
    sim.reset()
    b = sim.rollout(steps=3)
    assert [x.step for x in a] == [x.step for x in b]
