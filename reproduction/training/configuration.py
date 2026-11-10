"""Final E3/W7 configuration through existing research trainer hooks."""
import inspect
from types import SimpleNamespace
from . import checkpoints as base

WORKERS = 64
FRAGMENT = 256
BATCH = 16384

def configure(spec, env_name, graph, owner, worker_cpus, device):
    """Same PPO math and sampling geometry; explicit local CPU reservation."""
    from ray.rllib.algorithms.ppo import PPOConfig
    values = {k: v for k, v in spec.items() if k != "env"}
    values.update(hidden_dim=128, message_layers=3, num_lanes=4,
                  front_distance=30, back_distance=10,
                  model_preset=spec["model"].get("model_preset"))
    owner_args = SimpleNamespace(**values)
    config = graph._classic_api_stack(PPOConfig()).framework("torch")
    config = config.environment(env=env_name, env_config=spec["env"], disable_env_checking=False)
    config = graph._configure_workers(config, owner_args)
    parameters = inspect.signature(config.env_runners).parameters
    field = "num_cpus_per_env_runner"
    base.require(field in parameters or hasattr(config, field), "Ray lacks env runner CPU field")
    config = config.env_runners(**{
        field: worker_cpus,
        "num_gpus_per_env_runner": 0,
        "sample_timeout_s": 1800,
    })
    config = owner._configure_training(config, owner_args)
    config = config.training(kl_coeff=.2, kl_target=.01)
    config = config.resources(num_gpus=1.0 if device == "cuda" else 0.0,
                             num_cpus_for_main_process=1)
    config = config.callbacks(graph.PvTrainingMetricsCallbacks)
    config = config.debugging(seed=spec["seed"], log_level="WARN")
    raw = config.to_dict()
    checks = {"train_batch_size": BATCH, "gamma": .995, "lr": .0002, "clip_param": .2,
              "entropy_coeff": .01, "vf_loss_coeff": 1., "vf_clip_param": 10.,
              "kl_coeff": .2, "kl_target": .01, "rollout_fragment_length": FRAGMENT,
              "num_env_runners": WORKERS, "num_envs_per_env_runner": 1,
              field: worker_cpus, "num_gpus_per_env_runner": 0,
              "num_gpus": 1.0 if device == "cuda" else 0.0,
              "framework": "torch", "batch_mode": "truncate_episodes",
              "use_critic": True, "use_gae": True,
              "enable_rl_module_and_learner": False,
              "enable_env_runner_and_connector_v2": False}
    base.require(raw.get("sample_timeout_s") == 1800, "sample timeout not applied")
    for key, expected in checks.items():
        base.require(raw.get(key) == expected, f"effective config mismatch {key}: {raw.get(key)!r}")
    for names, expected in ((("lambda", "lambda_"), .95),
                            (("minibatch_size", "sgd_minibatch_size"), 1024),
                            (("num_epochs", "num_sgd_iter"), 6)):
        base.require(any(raw.get(k) == expected for k in names), f"config mismatch {names}")
    base.require(raw["env_config"] == spec["env"], "environment config mismatch")
    expected_model = ({"hidden_dim": 128, "message_layers": 3} if spec["architecture"] == "graph"
                      else {"model_preset": owner_args.model_preset, "num_lanes": 4,
                            "front_distance": 30, "back_distance": 10})
    base.require(raw["model"]["custom_model_config"] == expected_model, "model config mismatch")
    return config, raw

def set_driver_threads(torch, intra):
    torch.set_num_threads(intra)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        base.require(torch.get_num_interop_threads() == 1, "Torch interop pool already not 1")
    base.require(torch.get_num_threads() == intra, "Torch intra threads not applied")
    base.require(torch.get_num_interop_threads() == 1, "Torch interop threads not 1")

def worker_setup():
    """Runs in every Ray worker process: keep rollout-side Torch pools at 1/1."""
    import torch
    set_driver_threads(torch, 1)
