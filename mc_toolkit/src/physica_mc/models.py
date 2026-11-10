"""Lazy, standalone FP32 inference. Importing this module does not import torch."""
from collections.abc import Mapping
import os
import numpy as np

ARCHITECTURES = ('graph', 'node', 'grid')
_PRESETS = {'node': 'node-mlp-matched', 'grid': 'grid-cnn-matched'}
_MAX_WEIGHT_BYTES = 256 * 1024 * 1024


def normalize_config(architecture: str, config: Mapping) -> dict:
    if architecture not in ARCHITECTURES:
        raise ValueError(f'unsupported architecture: {architecture!r}')
    result = dict(config)
    defaults = dict(road_length=1000, num_lanes=4, front_distance=30,
                    back_distance=10, sensor_distance=60, cooldown_steps=1)
    for key, default in defaults.items():
        value = result.setdefault(key, default)
        minimum = 0 if key in ('front_distance', 'back_distance', 'cooldown_steps') else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    width = result['front_distance'] + result['back_distance'] + 1
    if width < 2 or width > result['road_length'] or result['sensor_distance'] >= result['road_length']:
        raise ValueError('invalid observation geometry')
    nodes = result['num_lanes'] * width
    if result.get('max_nodes', nodes) != nodes or nodes > 100_000:
        raise ValueError('invalid max_nodes')
    result['max_nodes'] = nodes
    if architecture == 'graph':
        for key, default in (('hidden_dim', 128), ('message_layers', 3)):
            value = result.setdefault(key, default)
            if type(value) is not int or not 1 <= value <= 1024:
                raise ValueError(f'invalid {key}')
        hidden, layers = result['hidden_dim'], result['message_layers']
        # Exact build_graph count, including biases, LayerNorm and value head.
        # Reject before importing Torch or allocating a randomly initialized model.
        parameters = 6 * hidden**2 + 62 * hidden + 4 + layers * (5 * hidden**2 + 9 * hidden)
        if parameters * 4 > _MAX_WEIGHT_BYTES:
            raise ValueError('graph FP32 parameters exceed size limit')
    else:
        preset = result.setdefault('model_preset', _PRESETS[architecture])
        if preset != _PRESETS[architecture]:
            raise ValueError(f'{architecture} requires {_PRESETS[architecture]}')
        hidden, layers = (128, 3) if architecture == 'node' else (80, 5)
        for key, value in (('hidden_dim', hidden), ('layer_count', layers)):
            if key in result and result[key] != value:
                raise ValueError(f'{preset} requires {key}={value}')
            result[key] = value
    return result


def configure_torch(device='cpu'):
    """Set deterministic process-wide torch flags; fail on unavailable devices."""
    # cuBLAS reads this before its first CUDA context. Never replace an incompatible setting.
    workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if workspace not in (':4096:8', ':16:8'):
        raise ValueError('incompatible CUBLAS_WORKSPACE_CONFIG')
    import torch
    target = torch.device(device)
    if target.type not in ('cpu', 'cuda') or (target.type == 'cpu' and target.index is not None):
        raise ValueError('only cpu and explicit CUDA devices are supported')
    if target.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        index = torch.cuda.current_device() if target.index is None else target.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError('CUDA device index out of range')
        target = torch.device('cuda', index)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    return torch, target


def validate_weights(weights: Mapping) -> dict[str, np.ndarray]:
    if not isinstance(weights, Mapping) or not weights or len(weights) > 4096:
        raise ValueError('expected a nonempty bounded weight mapping')
    arrays = {}
    size = 0
    for key, value in weights.items():
        if not isinstance(key, str) or not key or len(key) > 256 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.' for c in key):
            raise ValueError('unsafe weight key')
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype('float32'):
            raise ValueError(f'{key}: weights must be FP32 numeric arrays')
        if value.ndim > 4 or not np.isfinite(value).all():
            raise ValueError(f'{key}: invalid or nonfinite weights')
        size += value.nbytes
        if size > _MAX_WEIGHT_BYTES:
            raise ValueError('weights exceed size limit')
        arrays[key] = np.array(value, copy=True, order='C')
    return arrays


def build_model(architecture: str, config: Mapping, weights: Mapping | None = None, *, device='cpu'):
    config = normalize_config(architecture, config)
    arrays = None if weights is None else validate_weights(weights)
    torch, target = configure_torch(device)
    # Constructors are lazy; neither the public module nor metadata readers load torch.
    if architecture == 'graph':
        from .compat.graph import build_graph
        model = build_graph(config['hidden_dim'], config['message_layers'], config['max_nodes'])
    else:
        from .compat.nongraph import NodeMlpCore, GridCnnCore
        common = dict(node_dim=48, global_dim=4, max_nodes=config['max_nodes'], hidden_dim=config['hidden_dim'])
        if architecture == 'node':
            model = NodeMlpCore(**common, layer_count=3)
        else:
            model = GridCnnCore(**common, num_lanes=config['num_lanes'],
                                front_distance=config['front_distance'], back_distance=config['back_distance'],
                                dilations=(1, 2, 4, 8, 16))
    model = model.to(dtype=torch.float32, device=target)
    if arrays is not None:
        expected = model.state_dict()
        if set(arrays) != set(expected):
            raise ValueError(f'weight keys mismatch: missing={set(expected)-set(arrays)}, extra={set(arrays)-set(expected)}')
        for key, array in arrays.items():
            if tuple(array.shape) != tuple(expected[key].shape):
                raise ValueError(f'{key}: weight shape mismatch')
        model.load_state_dict({k: torch.from_numpy(v) for k, v in arrays.items()}, strict=True)
    model.eval()
    model.requires_grad_(False)
    model.mc_config = config
    model.mc_architecture = architecture
    return model


def infer_logits(model, observation: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return masked (max_nodes, 3) FP32 logits for one unbatched observation."""
    import torch
    device = next(model.parameters()).device
    configure_torch(device)
    n = model.mc_config['max_nodes']
    shapes = dict(node_features=(n, 48), node_mask=(n,), action_mask=(n, 3), global_features=(4,))
    if model.mc_architecture == 'graph':
        shapes.update(neighbor_index=(n, 6), neighbor_mask=(n, 6), edge_features=(n, 6, 3))
    tensors = {}
    for key, shape in shapes.items():
        array = np.asarray(observation[key])
        if array.shape != shape or array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
            raise ValueError(f'{key}: invalid observation shape/dtype/values')
        if key.endswith('mask') and not np.isin(array, (0, 1)).all():
            raise ValueError(f'{key}: mask must be binary')
        if key == 'neighbor_index' and (array.dtype.kind not in 'iu' or np.any(array < 0) or np.any(array >= n)):
            raise ValueError('neighbor_index out of bounds or noninteger')
        dtype = torch.int64 if key == 'neighbor_index' else torch.float32
        tensors[key] = torch.tensor(array, dtype=dtype, device=device).unsqueeze(0)
    if not np.asarray(observation['action_mask']).any(axis=-1).all():
        raise ValueError('every padded slot requires at least one legal action')
    if model.mc_architecture == 'grid':
        features = np.asarray(observation['node_features'])
        active = np.asarray(observation['node_mask']).astype(bool)
        cfg = model.mc_config
        positions = np.rint(features[:, 0] * np.where(features[:, 0] >= 0, cfg['front_distance'], cfg['back_distance']))
        columns = np.clip(positions + cfg['back_distance'], 0, cfg['front_distance'] + cfg['back_distance'])
        lanes = np.clip(np.rint(features[:, 1] * (cfg['num_lanes'] - 1)), 0, cfg['num_lanes'] - 1)
        cells = lanes * (cfg['front_distance'] + cfg['back_distance'] + 1) + columns
        if np.unique(cells[active]).size != int(active.sum()):
            raise ValueError('duplicate active grid cells would make scatter ambiguous')
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = model(tensors)
        logits = output[0] if isinstance(output, tuple) else output
    result = logits[0].cpu().numpy().copy()
    if result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError('nonfinite or non-FP32 logits')
    return result


def infer_action(model, observation: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.argmax(infer_logits(model, observation), axis=-1).astype(np.int64)
