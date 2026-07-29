#!/usr/bin/env python3
"""Train the centralized PV graph policy on the local machine with RLlib."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping


try:  # Keep parser/config unit tests usable without the optional Ray extra.
    from ray.rllib.algorithms.callbacks import DefaultCallbacks as _DefaultCallbacks
except ImportError:  # pragma: no cover - exercised in dependency-light installs
    try:  # Ray 2.5x public callback location.
        from ray.rllib.callbacks.callbacks import RLlibCallback as _DefaultCallbacks
    except ImportError:
        class _DefaultCallbacks:  # type: ignore[no-redef]
            pass


ENV_NAME = "snfs_centralized_pv_v0"
MODEL_NAME = "snfs_pv_graph_torch"


def _last_episode_info(episode: Any) -> Mapping[str, Any] | None:
    last_info_for = getattr(episode, "last_info_for", None)
    if callable(last_info_for):
        try:
            info = last_info_for()
        except (KeyError, TypeError):
            info = None
        if isinstance(info, Mapping):
            return info
    raw_infos = getattr(episode, "_last_infos", None)
    if isinstance(raw_infos, Mapping):
        for info in raw_infos.values():
            if isinstance(info, Mapping):
                return info
    return None


class PvTrainingMetricsCallbacks(_DefaultCallbacks):
    """Expose complete-episode PV speed grouped by sampled train condition."""

    def on_episode_end(self, *, episode, **kwargs) -> None:
        info = _last_episode_info(episode)
        if not info or "episode_mean_pv_speed" not in info:
            return
        condition = str(info.get("condition_key", "unknown"))
        step_count = max(1, int(info.get("step_index", 1)))
        pv_speed = float(info["episode_mean_pv_speed"])
        episode.custom_metrics["v_pv"] = pv_speed
        episode.custom_metrics[f"v_pv/{condition}"] = pv_speed
        episode.custom_metrics[f"applied_per_step/{condition}"] = (
            float(info.get("episode_applied_count", 0)) / step_count
        )
        episode.custom_metrics[f"rejected_per_step/{condition}"] = (
            float(info.get("episode_rejected_count", 0)) / step_count
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local factorized Graph-PPO training for one priority vehicle"
    )
    parser.add_argument("--road-length", type=int, default=1000)
    parser.add_argument("--num-lanes", type=int, default=4)
    parser.add_argument("--density", type=float, default=0.20)
    parser.add_argument("--av-fraction", type=float, default=0.95)
    parser.add_argument(
        "--train-densities",
        "--density-values",
        dest="train_densities",
        help="comma-separated reset grid; defaults to --density",
    )
    parser.add_argument(
        "--train-av-fractions",
        "--av-fraction-values",
        dest="train_av_fractions",
        help="comma-separated reset grid; defaults to --av-fraction",
    )
    parser.add_argument("--front-distance", type=int, default=30)
    parser.add_argument("--back-distance", type=int, default=10)
    parser.add_argument("--sensor-distance", type=int, default=60)
    parser.add_argument(
        "--cooldown-steps",
        type=int,
        default=1,
        help="blocked steps after an applied lane change; 0 disables cooldown",
    )
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument(
        "--warmup-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse one fully warmed HDV snapshot per density on each worker",
    )
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument(
        "--backend", choices=("auto", "reference", "optimized"), default="optimized"
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--applied-penalty", type=float, default=0.001)
    parser.add_argument("--rejected-penalty", type=float, default=0.002)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--vf-loss-coeff", type=float, default=1.0)
    parser.add_argument("--vf-clip-param", type=float, default=10.0)
    parser.add_argument("--rollout-fragment-length", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="local Ray rollout worker processes",
    )
    parser.add_argument("--envs-per-worker", type=int, default=1)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=0,
        help="0 disables the sampled-timestep stop condition",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument(
        "--restore-checkpoint",
        type=Path,
        help="checkpoint directory (or parent containing checkpoints) to resume",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/rl/pv_graph_ppo")
    )
    parser.add_argument(
        "--ray-local-mode",
        action="store_true",
        help="serial Ray debug mode; do not use for normal local training",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one tiny reference-backend iteration for integration checking",
    )
    return parser


def _classic_api_stack(config):
    if hasattr(config, "api_stack"):
        return config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    if hasattr(config, "experimental"):
        return config.experimental(_enable_new_api_stack=False)
    return config


def _configure_workers(config, args):
    if hasattr(config, "env_runners"):
        parameters = inspect.signature(config.env_runners).parameters
        kwargs = {"rollout_fragment_length": args.rollout_fragment_length}
        if "num_env_runners" in parameters:
            kwargs["num_env_runners"] = args.workers
        if "num_envs_per_env_runner" in parameters:
            kwargs["num_envs_per_env_runner"] = args.envs_per_worker
        if "batch_mode" in parameters:
            kwargs["batch_mode"] = "truncate_episodes"
        return config.env_runners(**kwargs)
    return config.rollouts(
        num_rollout_workers=args.workers,
        num_envs_per_worker=args.envs_per_worker,
        rollout_fragment_length=args.rollout_fragment_length,
        batch_mode="truncate_episodes",
    )


def _configure_training(config, args):
    parameters = inspect.signature(config.training).parameters
    kwargs = {
        "gamma": args.gamma,
        "lr": args.lr,
        "train_batch_size": args.train_batch_size,
        "clip_param": args.clip_param,
        "entropy_coeff": args.entropy_coeff,
        "vf_loss_coeff": args.vf_loss_coeff,
        "vf_clip_param": args.vf_clip_param,
        "model": {
            "custom_model": MODEL_NAME,
            "custom_model_config": {
                "hidden_dim": args.hidden_dim,
                "message_layers": args.message_layers,
            },
            "vf_share_layers": True,
        },
    }
    def training_field(*names: str) -> str:
        for name in names:
            if name in parameters or hasattr(config, name):
                return name
        joined = ", ".join(names)
        raise RuntimeError(
            f"RLlib PPOConfig exposes none of the expected training fields: {joined}"
        )

    # Recent RLlib releases moved these settings to AlgorithmConfig.training's
    # **kwargs, so they are attributes on PPOConfig but are absent from the
    # inspected PPOConfig.training signature. Prefer the current names, while
    # retaining support for older Ray releases.
    kwargs[training_field("lambda_", "lambda")] = args.gae_lambda
    kwargs[
        training_field("minibatch_size", "sgd_minibatch_size")
    ] = args.minibatch_size
    kwargs[training_field("num_epochs", "num_sgd_iter")] = args.epochs
    return config.training(**kwargs)


def _nested(result: dict, *paths, default=None):
    for path in paths:
        value = result
        found = True
        for key in path:
            if not isinstance(value, dict) or key not in value:
                found = False
                break
            value = value[key]
        if found:
            return value
    return default


def _checkpoint_path(save_result) -> str:
    checkpoint = getattr(save_result, "checkpoint", save_result)
    return str(getattr(checkpoint, "path", checkpoint))


def _parse_training_values(
    raw: str | None,
    *,
    fallback: float,
    name: str,
    allow_zero: bool,
) -> list[float]:
    values = (
        [float(part.strip()) for part in raw.split(",") if part.strip()]
        if raw is not None
        else [float(fallback)]
    )
    lower_ok = (lambda value: value >= 0.0) if allow_zero else (lambda value: value > 0.0)
    if not values or any(not lower_ok(value) or value > 1.0 for value in values):
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must contain comma-separated values in {interval}")
    return list(dict.fromkeys(values))


def _checkpoint_marker(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in ("algorithm_state.pkl", "rllib_checkpoint.json", ".is_checkpoint")
    )


def _resolve_checkpoint(raw_path: Path) -> Path:
    path = raw_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")
    if path.is_file():
        path = path.parent
    if _checkpoint_marker(path):
        return path
    candidates = {
        marker.parent
        for pattern in ("algorithm_state.pkl", "rllib_checkpoint.json", ".is_checkpoint")
        for marker in path.rglob(pattern)
    }
    if not candidates:
        raise FileNotFoundError(f"no RLlib checkpoint found below {path}")

    def rank(candidate: Path) -> tuple[int, float]:
        numbers = re.findall(r"\d+", candidate.name)
        sequence = int(numbers[-1]) if numbers else -1
        return sequence, candidate.stat().st_mtime

    return max(candidates, key=rank)


def _custom_metrics(result: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = _nested(
        dict(result),
        ("env_runners", "custom_metrics"),
        ("custom_metrics",),
        default={},
    )
    return metrics if isinstance(metrics, Mapping) else {}


def _metric_mean(metrics: Mapping[str, Any], name: str) -> float | None:
    for key in (f"{name}_mean", name):
        value = metrics.get(key)
        if isinstance(value, Mapping):
            value = value.get("mean")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _condition_metrics(
    result: Mapping[str, Any], metric_name: str
) -> dict[str, float]:
    metrics = _custom_metrics(result)
    prefix = f"{metric_name}/"
    output: dict[str, float] = {}
    for raw_key in metrics:
        key = str(raw_key)
        if not key.startswith(prefix):
            continue
        condition = key[len(prefix) :]
        if condition.endswith("_mean"):
            condition = condition[:-5]
        value = _metric_mean(metrics, f"{metric_name}/{condition}")
        if value is not None:
            output[condition] = value
    return dict(sorted(output.items()))


def _unique_checkpoint_dir(
    root: Path,
    *,
    label: str,
    iteration: int,
    sampled_steps: int,
) -> Path:
    stem = f"{label}_iter_{iteration:06d}_steps_{sampled_steps:012d}"
    candidate = root / stem
    suffix = 2
    while candidate.exists():
        candidate = root / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _save_unique_checkpoint(
    algorithm: Any,
    root: Path,
    *,
    label: str,
    iteration: int,
    sampled_steps: int,
) -> str:
    target = _unique_checkpoint_dir(
        root,
        label=label,
        iteration=iteration,
        sampled_steps=sampled_steps,
    )
    return _checkpoint_path(algorithm.save(str(target)))


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        args.road_length = 100
        args.density = 0.20
        args.train_densities = None
        args.train_av_fractions = None
        args.front_distance = 10
        args.back_distance = 5
        args.sensor_distance = 20
        args.warmup_steps = 5
        args.episode_steps = 32
        args.backend = "reference"
        args.workers = 0
        args.rollout_fragment_length = 32
        args.train_batch_size = 64
        args.minibatch_size = 32
        args.epochs = 1
        args.iterations = 1
        args.checkpoint_freq = 1

    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.cooldown_steps < 0:
        raise SystemExit("--cooldown-steps must be >= 0")
    try:
        density_values = _parse_training_values(
            args.train_densities,
            fallback=args.density,
            name="--train-densities",
            allow_zero=False,
        )
        if any(value >= 1.0 for value in density_values):
            raise ValueError("--train-densities values must be in (0, 1)")
        av_fraction_values = _parse_training_values(
            args.train_av_fractions,
            fallback=args.av_fraction,
            name="--train-av-fractions",
            allow_zero=True,
        )
        restore_checkpoint = (
            _resolve_checkpoint(args.restore_checkpoint)
            if args.restore_checkpoint is not None
            else None
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.models import ModelCatalog
        from ray.tune.registry import register_env

        from snfs_traffic.rl.centralized_pv_env import CentralizedPvEnv
        from snfs_traffic.rl.graph_ppo import FactorizedGraphPPO, PvGraphTorchModel
    except ImportError as exc:
        raise SystemExit(
            f"Missing local training dependency: {exc}\n"
            "Install once with: python -m pip install -e \".[rl,ray,numba]\""
        ) from exc

    env_config = {
        "road_length": args.road_length,
        "num_lanes": args.num_lanes,
        "density": args.density,
        "av_fraction": args.av_fraction,
        "density_values": density_values,
        "av_fraction_values": av_fraction_values,
        "front_distance": args.front_distance,
        "back_distance": args.back_distance,
        "sensor_distance": args.sensor_distance,
        "cooldown_steps": args.cooldown_steps,
        "warmup_steps": args.warmup_steps,
        "warmup_cache": args.warmup_cache,
        "episode_steps": args.episode_steps,
        "backend": args.backend,
        "validate": args.validate,
        "applied_penalty": args.applied_penalty,
        "rejected_penalty": args.rejected_penalty,
        "seed": args.seed,
    }
    register_env(ENV_NAME, lambda env_context: CentralizedPvEnv(env_context))
    ModelCatalog.register_custom_model(MODEL_NAME, PvGraphTorchModel)

    config = _classic_api_stack(PPOConfig())
    config = config.environment(
        env=ENV_NAME,
        env_config=env_config,
        disable_env_checking=False,
    )
    config = config.framework("torch")
    config = _configure_workers(config, args)
    config = _configure_training(config, args)
    if hasattr(config, "callbacks"):
        config = config.callbacks(PvTrainingMetricsCallbacks)
    if hasattr(config, "resources"):
        config = config.resources(num_gpus=args.num_gpus)
    if hasattr(config, "debugging"):
        config = config.debugging(seed=args.seed, log_level="WARN")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "train_metrics.jsonl"
    run_config = vars(args).copy()
    for path_key in ("out_dir", "restore_checkpoint"):
        if run_config.get(path_key) is not None:
            run_config[path_key] = str(run_config[path_key])
    run_config["resolved_restore_checkpoint"] = (
        str(restore_checkpoint) if restore_checkpoint is not None else None
    )
    run_config["env_config"] = env_config
    run_config["ray_version"] = ray.__version__
    with (args.out_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(run_config, stream, indent=2, sort_keys=True)

    ray.init(
        include_dashboard=False,
        local_mode=args.ray_local_mode,
        ignore_reinit_error=True,
        log_to_driver=True,
    )
    algorithm = None
    started = time.perf_counter()
    training_iteration = 0
    sampled_steps = 0
    try:
        algorithm = FactorizedGraphPPO(config=config)
        if restore_checkpoint is not None:
            algorithm.restore(str(restore_checkpoint))
            print(f"restored_checkpoint={restore_checkpoint}", flush=True)
        conditions = [
            f"rho={density:.3f}/av={fraction:.3f}"
            for density in density_values
            for fraction in av_fraction_values
        ]
        print(
            f"train_conditions={len(conditions)} " + ",".join(conditions),
            flush=True,
        )
        with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics_stream:
            for local_iteration in range(1, args.iterations + 1):
                result = algorithm.train()
                training_iteration = int(
                    result.get("training_iteration", local_iteration)
                    or local_iteration
                )
                sampled_steps = int(
                    _nested(
                        result,
                        ("num_env_steps_sampled_lifetime",),
                        ("num_env_steps_sampled",),
                        ("timesteps_total",),
                        default=0,
                    )
                    or 0
                )
                episode_return = _nested(
                    result,
                    ("env_runners", "episode_return_mean"),
                    ("episode_reward_mean",),
                    default=None,
                )
                learner_loss = _nested(
                    result,
                    ("learners", "default_policy", "learner_stats", "total_loss"),
                    ("info", "learner", "default_policy", "learner_stats", "total_loss"),
                    default=None,
                )
                custom_metrics = _custom_metrics(result)
                pv_speed_mean = _metric_mean(custom_metrics, "v_pv")
                pv_speed_by_condition = _condition_metrics(result, "v_pv")
                applied_by_condition = _condition_metrics(
                    result, "applied_per_step"
                )
                rejected_by_condition = _condition_metrics(
                    result, "rejected_per_step"
                )
                elapsed = time.perf_counter() - started
                compact = {
                    "iteration": training_iteration,
                    "local_iteration": local_iteration,
                    "sampled_env_steps": sampled_steps,
                    "episode_return_mean": episode_return,
                    "pv_speed_mean": pv_speed_mean,
                    "pv_speed_by_condition": pv_speed_by_condition,
                    "applied_per_step_by_condition": applied_by_condition,
                    "rejected_per_step_by_condition": rejected_by_condition,
                    "total_loss": learner_loss,
                    "elapsed_seconds": elapsed,
                }
                metrics_stream.write(json.dumps(compact, sort_keys=True) + "\n")
                condition_text = (
                    ",".join(
                        f"{key}={value:.3f}"
                        for key, value in pv_speed_by_condition.items()
                    )
                    or "waiting-for-complete-episodes"
                )
                print(
                    f"[{local_iteration}/{args.iterations}] "
                    f"train_iter={training_iteration} steps={sampled_steps} "
                    f"return={episode_return} v_pv={pv_speed_mean} "
                    f"loss={learner_loss} elapsed={elapsed:.1f}s "
                    f"v_pv_by_condition=[{condition_text}]",
                    flush=True,
                )
                if (
                    args.checkpoint_freq > 0
                    and training_iteration % args.checkpoint_freq == 0
                ):
                    saved = _save_unique_checkpoint(
                        algorithm,
                        checkpoint_dir,
                        label="periodic",
                        iteration=training_iteration,
                        sampled_steps=sampled_steps,
                    )
                    print(f"checkpoint={saved}", flush=True)
                if args.stop_timesteps > 0 and sampled_steps >= args.stop_timesteps:
                    break
        saved = _save_unique_checkpoint(
            algorithm,
            checkpoint_dir,
            label="final",
            iteration=training_iteration,
            sampled_steps=sampled_steps,
        )
        print(f"final_checkpoint={saved}", flush=True)
    except KeyboardInterrupt:
        if algorithm is not None:
            saved = _save_unique_checkpoint(
                algorithm,
                checkpoint_dir,
                label="interrupted",
                iteration=training_iteration,
                sampled_steps=sampled_steps,
            )
            print(f"interrupted_checkpoint={saved}", flush=True)
        return 130
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
