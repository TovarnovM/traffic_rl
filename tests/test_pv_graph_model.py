from __future__ import annotations

import numpy as np
import pytest


gymnasium = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")
pytest.importorskip("ray")

from snfs_traffic.rl.centralized_pv_env import CentralizedPvEnv  # noqa: E402
from snfs_traffic.rl.graph_ppo import PvGraphTorchModel  # noqa: E402


def test_graph_model_emits_one_masked_categorical_head_per_slot_and_one_value():
    env = CentralizedPvEnv(
        {
            "num_lanes": 4,
            "road_length": 100,
            "density": 0.10,
            "av_fraction": 1.0,
            "front_distance": 10,
            "back_distance": 5,
            "sensor_distance": 20,
            "warmup_steps": 0,
            "episode_steps": 2,
            "backend": "reference",
            "seed": 5,
        }
    )
    observation, _info = env.reset(seed=5)
    model = PvGraphTorchModel(
        env.observation_space,
        env.action_space,
        int(np.sum(env.action_space.nvec)),
        {
            "custom_model_config": {"hidden_dim": 32, "message_layers": 2},
            "_disable_preprocessor_api": True,
        },
        "test_pv_graph_model",
    )
    torch_observation = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in observation.items()
    }

    logits, _state = model.forward(
        {"obs": torch_observation}, [], None
    )
    logits = logits.reshape(1, env.action_space.shape[0], 3)

    assert logits.shape == (1, env.action_space.shape[0], 3)
    assert model.value_function().shape == (1,)
    invalid = torch.from_numpy(observation["action_mask"] == 0).unsqueeze(0)
    assert torch.all(logits[invalid] < -1.0e8)
    assert torch.isfinite(model.value_function()).all()
