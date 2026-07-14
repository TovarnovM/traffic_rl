#!/usr/bin/env python3
"""Train the centralized PV graph policy on the local machine with RLlib."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path


ENV_NAME = "snfs_centralized_pv_v0"
MODEL_NAME = "snfs_pv_graph_torch"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local factorized Graph-PPO training for one priority vehicle"
    )
    parser.add_argument("--road-length", type=int, default=1000)
    parser.add_argument("--num-lanes", type=int, default=4)
    parser.add_argument("--density", type=float, default=0.20)
    parser.add_argument("--av-fraction", type=float, default=0.95)
    parser.add_argument("--front-distance", type=int, default=30)
    parser.add_argument("--back-distance", type=int, default=10)
    parser.add_argument("--sensor-distance", type=int, default=60)
    parser.add_argument("--cooldown-steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
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
    if "lambda_" in parameters:
        kwargs["lambda_"] = args.gae_lambda
    elif "lambda" in parameters:  # pragma: no cover - future compatibility
        kwargs["lambda"] = args.gae_lambda
    if "train_batch_size" in parameters:
        kwargs["train_batch_size"] = args.train_batch_size
    if "sgd_minibatch_size" in parameters:
        kwargs["sgd_minibatch_size"] = args.minibatch_size
    elif "minibatch_size" in parameters:
        kwargs["minibatch_size"] = args.minibatch_size
    if "num_sgd_iter" in parameters:
        kwargs["num_sgd_iter"] = args.epochs
    elif "num_epochs" in parameters:
        kwargs["num_epochs"] = args.epochs
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


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        args.road_length = 100
        args.density = 0.20
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
        "front_distance": args.front_distance,
        "back_distance": args.back_distance,
        "sensor_distance": args.sensor_distance,
        "cooldown_steps": args.cooldown_steps,
        "warmup_steps": args.warmup_steps,
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
    if hasattr(config, "resources"):
        config = config.resources(num_gpus=args.num_gpus)
    if hasattr(config, "debugging"):
        config = config.debugging(seed=args.seed, log_level="WARN")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "train_metrics.jsonl"
    run_config = vars(args).copy()
    run_config["out_dir"] = str(run_config["out_dir"])
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
    try:
        algorithm = FactorizedGraphPPO(config=config)
        with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics_stream:
            for iteration in range(1, args.iterations + 1):
                result = algorithm.train()
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
                elapsed = time.perf_counter() - started
                compact = {
                    "iteration": iteration,
                    "sampled_env_steps": sampled_steps,
                    "episode_return_mean": episode_return,
                    "total_loss": learner_loss,
                    "elapsed_seconds": elapsed,
                }
                metrics_stream.write(json.dumps(compact, sort_keys=True) + "\n")
                print(
                    f"[{iteration}/{args.iterations}] steps={sampled_steps} "
                    f"return={episode_return} loss={learner_loss} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                if args.checkpoint_freq > 0 and iteration % args.checkpoint_freq == 0:
                    saved = algorithm.save(str(checkpoint_dir))
                    print(f"checkpoint={_checkpoint_path(saved)}", flush=True)
                if args.stop_timesteps > 0 and sampled_steps >= args.stop_timesteps:
                    break
        saved = algorithm.save(str(checkpoint_dir))
        print(f"final_checkpoint={_checkpoint_path(saved)}", flush=True)
    except KeyboardInterrupt:
        if algorithm is not None:
            saved = algorithm.save(str(checkpoint_dir))
            print(f"interrupted_checkpoint={_checkpoint_path(saved)}", flush=True)
        return 130
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
