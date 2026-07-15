from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_runner():
    research = Path(__file__).resolve().parents[1] / "research"
    if str(research) not in sys.path:
        sys.path.insert(0, str(research))
    path = research / "train_pv_nongraph_ppo.py"
    spec = importlib.util.spec_from_file_location("pv_nongraph_train_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _TrainingConfig:
    def __init__(self) -> None:
        self.lambda_ = None
        self.minibatch_size = 128
        self.num_epochs = 30
        self.received = {}

    def training(self, **kwargs):
        self.received = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)
        return self


def _args():
    return SimpleNamespace(
        gamma=0.995,
        lr=2e-4,
        train_batch_size=16_384,
        clip_param=0.2,
        entropy_coeff=0.01,
        vf_loss_coeff=1.0,
        vf_clip_param=10.0,
        gae_lambda=0.95,
        minibatch_size=1024,
        epochs=6,
        model_preset="grid-cnn-matched",
        num_lanes=4,
        front_distance=30,
        back_distance=10,
    )


def test_training_config_keeps_factorized_ppo_settings_and_geometry():
    runner = _load_runner()
    config = _TrainingConfig()

    configured = runner._configure_training(config, _args())

    assert configured is config
    assert config.received["train_batch_size"] == 16_384
    assert config.received["minibatch_size"] == 1024
    assert config.received["num_epochs"] == 6
    custom = config.received["model"]["custom_model_config"]
    assert custom == {
        "model_preset": "grid-cnn-matched",
        "num_lanes": 4,
        "front_distance": 30,
        "back_distance": 10,
    }


def test_parser_exposes_only_frozen_publication_presets():
    runner = _load_runner()
    args = runner.build_parser().parse_args([])

    assert runner.MODEL_PRESETS == (
        "node-mlp-matched",
        "grid-cnn-matched",
        "grid-cnn-large",
    )
    assert args.model_preset == "node-mlp-matched"
    assert args.cooldown_steps == 1
    assert args.warmup_cache is True

