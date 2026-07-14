from __future__ import annotations

import pytest


def test_multiagent_environments_pass_rllib_precheck_when_ray_is_installed():
    pytest.importorskip("ray")
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    from ray.rllib.utils.pre_checks.env import check_multiagent_environments

    from snfs_traffic.rl.multiagent_env import SnfsTrafficMultiAgentEnv
    from snfs_traffic.rl.speed_control_multiagent_env import (
        SnfsTrafficSpeedControlMultiAgentEnv,
    )

    for env_cls in (SnfsTrafficMultiAgentEnv, SnfsTrafficSpeedControlMultiAgentEnv):
        assert issubclass(env_cls, MultiAgentEnv)
        env = env_cls(backend="reference", num_controlled=3, density=0.2, seed=123)
        check_multiagent_environments(env)
