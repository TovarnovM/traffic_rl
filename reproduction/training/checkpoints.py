"""Scientific identity and functional checkpoint helpers extracted from e3_gpu_pilot.

Portable release ownership; no sampler/environment/RNG-cache resume guarantee.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

OWNER = "physica-a-local-v1.1"
BATCH = 16384
COUNTS = {"graph": 355460, "node": 354308, "grid": 353124}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def scientific_config(architecture, seed):
    require(architecture in COUNTS, "unknown architecture")
    return {
        "architecture": architecture, "seed": seed, "parameters": COUNTS[architecture],
        "model": ({"hidden_dim": 128, "message_layers": 3} if architecture == "graph"
                  else {"model_preset": "node-mlp-matched" if architecture == "node" else "grid-cnn-matched"}),
        "workers": 64, "envs_per_worker": 1, "rollout_fragment_length": 256,
        "train_batch_size": BATCH, "minibatch_size": 1024, "epochs": 6,
        "lr": .0002, "gamma": .995, "gae_lambda": .95, "clip_param": .2,
        "entropy_coeff": .01, "vf_loss_coeff": 1., "vf_clip_param": 10.,
        "kl_coeff": .2, "kl_target": .01, "dtype": "float32", "amp": False,
        "tf32_matmul": False, "tf32_cudnn": False,
        "env": {"road_length": 1000, "num_lanes": 4, "density": .2,
                "av_fraction": .95, "density_values": [.1, .2, .3, .4],
                "av_fraction_values": [.9, .95, 1.], "front_distance": 30,
                "back_distance": 10, "sensor_distance": 60, "cooldown_steps": 1,
                "episode_steps": 1000, "warmup_steps": 1000, "warmup_cache": True,
                "backend": "optimized", "validate": False,
                "applied_penalty": .001, "rejected_penalty": .002, "seed": seed}}

def stable_digest(value):
    """Typed, length-framed digest; tensor device does not affect value identity."""
    h = hashlib.sha256()

    def emit(tag, data=b""):
        h.update(tag.encode() + b":" + str(len(data)).encode() + b":" + data)

    def visit(v):
        if hasattr(v, "detach"):
            v = v.detach().cpu().contiguous().numpy()
        if hasattr(v, "dtype") and hasattr(v, "shape") and hasattr(v, "tobytes"):
            require(not getattr(v.dtype, "hasobject", False), "object arrays unsupported")
            emit("array", str(v.dtype).encode()); visit(tuple(v.shape))
            emit("bytes", v.tobytes(order="C"))
        elif isinstance(v, dict):
            emit("dict")
            for key in sorted(v, key=lambda k: (type(k).__name__, repr(k))):
                visit(key); visit(v[key])
        elif isinstance(v, (list, tuple)):
            emit(type(v).__name__, str(len(v)).encode())
            for item in v:
                visit(item)
        elif v is None or isinstance(v, (str, bool, int, float, bytes)):
            emit(type(v).__name__, v if isinstance(v, bytes) else repr(v).encode())
        elif hasattr(v, "item"):
            visit(v.item())
        else:
            raise TypeError(f"unsupported digest type: {type(v)}")
    visit(value)
    return h.hexdigest()

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def json_write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())

def within(path, root):
    resolved, base = path.resolve(), root.resolve()
    require(resolved != base and resolved.is_relative_to(base), "not an owned child path")
    require(not path.is_symlink(), "symlink forbidden")
    return resolved

def file_hashes(root):
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "checkpoint symlink forbidden")
        if path.is_file() and path != root / "COMPLETE.json":
            within(path, root)
            result[path.relative_to(root).as_posix()] = sha(path)
    return result

def identity(config, sources):
    return {"owner": OWNER, "architecture": config["architecture"], "seed": config["seed"],
            "config_digest": stable_digest(config), "source_digest": stable_digest(sources)}

def validate_checkpoint(path, expected):
    path = path.absolute()
    require(path == path.resolve(), "checkpoint path must be resolved and nonsymlinked")
    require(path.parent.name == "checkpoints", "not an owned checkpoint directory")
    phase = path.parent.parent
    owner = json.loads((phase / "run_config.json").read_text())
    require(owner["identity"] == expected, "checkpoint phase identity mismatch")
    manifest = json.loads((path / "COMPLETE.json").read_text())
    require(manifest["identity"] == expected, "checkpoint source/config/arch/seed mismatch")
    require(manifest["files"] and file_hashes(path) == manifest["files"],
            "checkpoint missing, extra or tampered files")
    payload = within(path / manifest["payload"], path)
    require(payload.is_dir(), "missing Algorithm payload")
    require(manifest["snapshot"]["iteration"] >= 1, "invalid checkpoint iteration")
    require(path.name == f"iter_{manifest['snapshot']['iteration']:06d}", "unpublished checkpoint")
    require(manifest["snapshot"]["sampled"] == manifest["snapshot"]["iteration"] * BATCH,
            "checkpoint sampled counter mismatch")
    require(manifest["snapshot"]["trained"] == manifest["snapshot"]["sampled"],
            "checkpoint trained counter mismatch")
    return manifest, payload

def publish_checkpoint(algorithm, root, expected, snapshot, save_helper):
    """Algorithm.save via frozen owner's unique helper, then validated atomic publish."""
    root = root.resolve()
    staging = root / (".staging-" + uuid.uuid4().hex)
    within(staging, root).mkdir()
    payload = Path(save_helper(algorithm, staging, label="algorithm",
                              iteration=snapshot["iteration"], sampled_steps=snapshot["sampled"]))
    payload = within(payload, staging)
    manifest = {"identity": expected, "snapshot": snapshot,
                "payload": payload.relative_to(staging).as_posix(), "files": file_hashes(staging)}
    require(manifest["files"], "empty checkpoint")
    json_write(staging / "COMPLETE.json", manifest)
    require(file_hashes(staging) == manifest["files"], "checkpoint changed during save")
    target = root / f"iter_{snapshot['iteration']:06d}"
    within(target, root); within(staging, root)
    require(not target.exists(), "immutable checkpoint already exists")
    os.rename(staging, target)
    pointer = root / (".LATEST-" + uuid.uuid4().hex)
    json_write(pointer, {"checkpoint": target.name, "manifest_sha256": sha(target / "COMPLETE.json")})
    within(pointer, root); within(root / "LATEST", root)
    os.replace(pointer, root / "LATEST")
    return target

def assert_resume(actual, saved):
    require(actual == saved, "restore weights/optimizer/LR/KL/counters digest mismatch")

def check_steps(result, previous):
    iteration = int(result["training_iteration"])
    sampled = result.get("num_env_steps_sampled_lifetime", result.get("num_env_steps_sampled", result.get("timesteps_total")))
    trained = result.get("num_env_steps_trained_lifetime", result.get("num_env_steps_trained"))
    require(sampled is not None and trained is not None, "missing lifetime env counters")
    require(iteration == previous + 1 and int(sampled) == iteration * BATCH
            and int(trained) == iteration * BATCH, "iteration/lifetime step mismatch")
    for key in ("num_env_steps_sampled_this_iter", "timesteps_this_iter", "num_env_steps_trained_this_iter"):
        if key in result:
            require(int(result[key]) == BATCH, "iteration step mismatch")
    return iteration

def algorithm_snapshot(algorithm):
    policy = algorithm.get_policy()
    named = {id(p): n for n, p in policy.model.named_parameters()}
    optimizers = []
    for optimizer in policy._optimizers:
        groups = []
        for group in optimizer.param_groups:
            groups.append({**{k: v for k, v in group.items() if k != "params"},
                           "params": [named[id(p)] for p in group["params"]]})
        optimizers.append({"groups": groups, "state": {
            named[id(p)]: value for p, value in optimizer.state.items()}})
    require(optimizers and any(o["state"] for o in optimizers), "no optimizer moments")
    counters = dict(algorithm._counters)
    sampled = int(counters["num_env_steps_sampled"])
    trained = int(counters["num_env_steps_trained"])
    fields = {key: getattr(policy, key) for key in
              ("cur_lr", "kl_coeff", "entropy_coeff", "global_timestep")}
    return {"iteration": int(algorithm.iteration), "sampled": sampled, "trained": trained,
            "weights": stable_digest(policy.get_weights()), "optimizer": stable_digest(optimizers),
            "lr_kl": stable_digest(fields), "counters": stable_digest(counters)}

def scalar_tree(value):
    if isinstance(value, dict):
        return {str(k): scalar_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar_tree(v) for v in value]
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        value = value.item()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    return {"type": type(value).__name__}
