from __future__ import annotations

from snfs_traffic.rl.episode import EpisodeConfig, build_reset_info, build_step_info, compute_episode_flags
from snfs_traffic.rl.rewards import RewardConfig, compute_controlled_reward

__all__ = [
    "EpisodeConfig",
    "RewardConfig",
    "SnfsTrafficEnv",
    "SnfsTrafficMultiAgentEnv",
    "SnfsTrafficSpeedControlMultiAgentEnv",
    "build_reset_info",
    "build_step_info",
    "compute_controlled_reward",
    "compute_episode_flags",
]


def __getattr__(name: str):
    if name == "SnfsTrafficEnv":
        from snfs_traffic.rl.env import SnfsTrafficEnv

        return SnfsTrafficEnv
    if name == "SnfsTrafficMultiAgentEnv":
        from snfs_traffic.rl.multiagent_env import SnfsTrafficMultiAgentEnv

        return SnfsTrafficMultiAgentEnv
    if name == "SnfsTrafficSpeedControlMultiAgentEnv":
        from snfs_traffic.rl.speed_control_multiagent_env import SnfsTrafficSpeedControlMultiAgentEnv

        return SnfsTrafficSpeedControlMultiAgentEnv
    raise AttributeError(name)
