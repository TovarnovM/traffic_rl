from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


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


def test_mixed_training_grid_parser_and_new_defaults():
    runner = _load_runner()
    args = runner.build_parser().parse_args([])

    assert args.cooldown_steps == 1
    assert args.warmup_cache is True
    assert runner._parse_training_values(
        "0.10,0.30,0.10",
        fallback=0.20,
        name="rho",
        allow_zero=False,
    ) == [0.10, 0.30]
    assert runner._parse_training_values(
        "0,0.95,1",
        fallback=0.95,
        name="av",
        allow_zero=True,
    ) == [0.0, 0.95, 1.0]
    with pytest.raises(ValueError):
        runner._parse_training_values(
            "0,0.20",
            fallback=0.20,
            name="rho",
            allow_zero=False,
        )


def test_condition_metrics_are_extracted_from_rllib_custom_metrics():
    runner = _load_runner()
    result = {
        "custom_metrics": {
            "v_pv_mean": 4.25,
            "v_pv/rho_0p200_av_0p950_mean": 4.10,
            "v_pv/rho_0p300_av_1p000_mean": 4.40,
        }
    }

    metrics = runner._custom_metrics(result)
    assert runner._metric_mean(metrics, "v_pv") == pytest.approx(4.25)
    assert runner._condition_metrics(result, "v_pv") == {
        "rho_0p200_av_0p950": pytest.approx(4.10),
        "rho_0p300_av_1p000": pytest.approx(4.40),
    }


def test_callback_records_complete_episode_speed_and_action_rates():
    runner = _load_runner()

    class Episode:
        custom_metrics = {}

        @staticmethod
        def last_info_for():
            return {
                "condition_key": "rho_0p300_av_0p950",
                "episode_mean_pv_speed": 4.5,
                "episode_applied_count": 20,
                "episode_rejected_count": 10,
                "step_index": 100,
            }

    episode = Episode()
    runner.PvTrainingMetricsCallbacks().on_episode_end(episode=episode)

    assert episode.custom_metrics["v_pv"] == pytest.approx(4.5)
    assert episode.custom_metrics["v_pv/rho_0p300_av_0p950"] == pytest.approx(4.5)
    assert episode.custom_metrics[
        "applied_per_step/rho_0p300_av_0p950"
    ] == pytest.approx(0.2)
    assert episode.custom_metrics[
        "rejected_per_step/rho_0p300_av_0p950"
    ] == pytest.approx(0.1)


def test_checkpoint_directories_are_unique_and_iteration_named(tmp_path):
    runner = _load_runner()
    first = runner._unique_checkpoint_dir(
        tmp_path,
        label="periodic",
        iteration=25,
        sampled_steps=102_400,
    )
    second = runner._unique_checkpoint_dir(
        tmp_path,
        label="periodic",
        iteration=25,
        sampled_steps=102_400,
    )

    assert first.name == "periodic_iter_000025_steps_000000102400"
    assert second.name == "periodic_iter_000025_steps_000000102400_02"
