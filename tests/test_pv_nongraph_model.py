from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")
pytest.importorskip("ray")

from snfs_traffic.rl.centralized_pv_env import CentralizedPvEnv  # noqa: E402
from ray.rllib.models.preprocessors import get_preprocessor  # noqa: E402
from snfs_traffic.rl.nongraph_ppo import (  # noqa: E402
    GRID_CNN_LARGE,
    GRID_CNN_MATCHED,
    NODE_MLP_MATCHED,
    PvNonGraphTorchModel,
    count_trainable_parameters,
)


EXPECTED_PARAMETERS = {
    NODE_MLP_MATCHED: 354_308,
    GRID_CNN_MATCHED: 353_124,
    GRID_CNN_LARGE: 1_390_596,
}


def _env():
    return CentralizedPvEnv(
        {
            "num_lanes": 4,
            "road_length": 100,
            "density": 0.20,
            "av_fraction": 1.0,
            "front_distance": 30,
            "back_distance": 10,
            "sensor_distance": 40,
            "cooldown_steps": 1,
            "warmup_steps": 0,
            "episode_steps": 2,
            "backend": "reference",
            "seed": 5,
        }
    )


def _model(env, preset: str):
    return PvNonGraphTorchModel(
        env.observation_space,
        env.action_space,
        int(np.sum(env.action_space.nvec)),
        {
            "custom_model_config": {
                "model_preset": preset,
                "num_lanes": 4,
                "front_distance": 30,
                "back_distance": 10,
            },
        },
        f"test_{preset}",
    )


@pytest.mark.parametrize("preset", list(EXPECTED_PARAMETERS))
def test_non_graph_models_match_declared_size_and_factorized_contract(preset):
    env = _env()
    observation, _info = env.reset(seed=5)
    model = _model(env, preset)
    tensors = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in observation.items()
    }

    logits, _state = model.forward({"obs": tensors}, [], None)
    reshaped = logits.reshape(1, env.action_space.shape[0], 3)

    assert count_trainable_parameters(model) == EXPECTED_PARAMETERS[preset]
    assert reshaped.shape == (1, env.action_space.shape[0], 3)
    assert model.value_function().shape == (1,)
    invalid = torch.from_numpy(observation["action_mask"] == 0).unsqueeze(0)
    assert torch.all(reshaped[invalid] < -1.0e8)
    assert torch.isfinite(model.value_function()).all()


@pytest.mark.parametrize(
    "preset", [NODE_MLP_MATCHED, GRID_CNN_MATCHED, GRID_CNN_LARGE]
)
def test_non_graph_logits_do_not_depend_on_dynamic_graph_tensors(preset):
    env = _env()
    observation, _info = env.reset(seed=9)
    model = _model(env, preset).eval()

    def forward(value):
        tensors = {
            key: torch.from_numpy(array).unsqueeze(0)
            for key, array in value.items()
        }
        with torch.inference_mode():
            logits, _state = model.forward({"obs": tensors}, [], None)
        return logits

    original = forward(observation)
    changed = {key: value.copy() for key, value in observation.items()}
    rng = np.random.default_rng(17)
    changed["neighbor_index"] = rng.integers(
        0,
        env.action_space.shape[0],
        size=changed["neighbor_index"].shape,
        dtype=changed["neighbor_index"].dtype,
    )
    changed["neighbor_mask"] = 1 - changed["neighbor_mask"]
    changed["edge_features"] = rng.uniform(
        -1.0, 1.0, size=changed["edge_features"].shape
    ).astype(np.float32)

    torch.testing.assert_close(original, forward(changed), rtol=0.0, atol=0.0)


def test_grid_cnn_raster_contains_one_cell_per_active_av():
    env = _env()
    observation, _info = env.reset(seed=13)
    model = _model(env, GRID_CNN_MATCHED)
    core = model.core
    node_features = torch.from_numpy(observation["node_features"]).unsqueeze(0)
    node_mask = torch.from_numpy(observation["node_mask"]).unsqueeze(0).float()
    lane, column = core._slot_coordinates(node_features)
    raster = core._rasterize(node_features, node_mask, lane, column)

    occupancy = raster[:, -1]
    assert int(occupancy.sum().item()) == int(observation["node_mask"].sum())
    assert occupancy.shape == (1, 4, 41)


def test_non_graph_wrapper_restores_flat_rllib_observation():
    env = _env()
    observation, _info = env.reset(seed=21)
    model = _model(env, GRID_CNN_MATCHED)
    preprocessor = get_preprocessor(env.observation_space)(env.observation_space)
    flat = preprocessor.transform(observation)

    logits, _state = model.forward(
        {"obs": torch.from_numpy(flat).unsqueeze(0)}, [], None
    )

    assert logits.shape == (1, int(np.sum(env.action_space.nvec)))
