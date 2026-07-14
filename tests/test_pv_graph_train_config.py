from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "train_pv_graph_ppo.py"
    )
    spec = importlib.util.spec_from_file_location("pv_graph_train_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _TrainingConfig:
    def __init__(self, *, modern: bool) -> None:
        self.lambda_ = None
        if modern:
            self.minibatch_size = 128
            self.num_epochs = 30
        else:
            self.sgd_minibatch_size = 128
            self.num_sgd_iter = 30
        self.received = {}

    def training(self, **kwargs):
        self.received = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)
        return self


def _args():
    return SimpleNamespace(
        gamma=0.995,
        lr=3e-4,
        train_batch_size=64,
        clip_param=0.2,
        entropy_coeff=0.01,
        vf_loss_coeff=1.0,
        vf_clip_param=10.0,
        hidden_dim=64,
        message_layers=2,
        gae_lambda=0.95,
        minibatch_size=32,
        epochs=1,
    )


def test_training_config_uses_current_rllib_fields_hidden_behind_kwargs():
    runner = _load_runner()
    config = _TrainingConfig(modern=True)

    configured = runner._configure_training(config, _args())

    assert configured is config
    assert config.received["train_batch_size"] == 64
    assert config.received["minibatch_size"] == 32
    assert config.received["num_epochs"] == 1
    assert "sgd_minibatch_size" not in config.received
    assert "num_sgd_iter" not in config.received


def test_training_config_keeps_legacy_rllib_field_names_as_fallback():
    runner = _load_runner()
    config = _TrainingConfig(modern=False)

    runner._configure_training(config, _args())

    assert config.received["sgd_minibatch_size"] == 32
    assert config.received["num_sgd_iter"] == 1
