import numpy as np
import pytest
from physica_mc.models import build_model, infer_action, infer_logits, normalize_config


@pytest.fixture(params=['graph', 'node', 'grid'])
def architecture(request):
    return request.param


def observation(n=164):
    features = np.zeros((n, 48), np.float32)
    features[0, 0], features[1, 0] = -1, 1
    mask = np.zeros(n, np.int8)
    mask[:2] = 1
    actions = np.zeros((n, 3), np.int8)
    actions[:, 1] = 1
    actions[0] = (1, 1, 0)
    return dict(node_features=features, node_mask=mask, action_mask=actions,
                global_features=np.zeros(4, np.float32),
                neighbor_index=np.zeros((n, 6), np.int32),
                neighbor_mask=np.zeros((n, 6), np.int8),
                edge_features=np.zeros((n, 6, 3), np.float32))


def weights_of(model):
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def test_deterministic_masked_strict_roundtrip(architecture):
    model = build_model(architecture, {})
    obs = observation()
    first = infer_logits(model, obs)
    np.testing.assert_array_equal(first, infer_logits(model, obs))
    restored = build_model(architecture, {}, weights_of(model))
    np.testing.assert_array_equal(first, infer_logits(restored, obs))
    actions = infer_action(model, obs)
    assert actions.dtype == np.int64 and actions.shape == (164,)
    assert obs['action_mask'][np.arange(164), actions].all()
    assert (actions[2:] == 1).all()
    assert first.dtype == np.float32 and np.isfinite(first).all()
    assert (first[obs['action_mask'] == 0] == np.float32(-1e9)).all()
    assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_checkpoint_parameter_contract(architecture):
    model = build_model(architecture, {})
    expected = {'graph': 355460, 'node': 354308, 'grid': 353124}
    assert sum(p.numel() for p in model.parameters()) == expected[architecture]
    weights = weights_of(model)
    assert 'value_head.2.weight' in weights
    assert all(not k.startswith('core.') for k in weights)


def test_known_logits_and_ties(architecture):
    original = build_model(architecture, {})
    weights = {k: np.zeros_like(v) for k, v in weights_of(original).items()}
    model = build_model(architecture, {}, weights)
    obs = observation()
    np.testing.assert_array_equal(infer_logits(model, obs),
                                  np.where(obs['action_mask'], np.float32(0), np.float32(-1e9)))
    assert infer_action(model, obs)[0] == 0


@pytest.mark.parametrize('damage', ['missing', 'extra', 'shape', 'nan', 'inf', 'float64', 'object'])
def test_strict_weights(architecture, damage):
    weights = weights_of(build_model(architecture, {}))
    key = next(iter(weights))
    if damage == 'missing':
        del weights[key]
    elif damage == 'extra':
        weights['bogus'] = np.zeros(1, np.float32)
    elif damage == 'shape':
        weights[key] = weights[key].reshape(-1)
    elif damage in ('nan', 'inf'):
        weights[key].flat[0] = float(damage)
    else:
        weights[key] = weights[key].astype(damage)
    with pytest.raises(ValueError):
        build_model(architecture, {}, weights)


def test_invalid_observation(architecture):
    model = build_model(architecture, {})
    obs = observation()
    obs['action_mask'][0] = 0
    with pytest.raises(ValueError, match='legal action'):
        infer_action(model, obs)
    obs = observation()
    obs['node_features'][0, 0] = np.nan
    with pytest.raises(ValueError):
        infer_logits(model, obs)


def test_graph_neighbor_bounds():
    model = build_model('graph', {})
    obs = observation()
    obs['neighbor_index'][0, 0] = -1
    with pytest.raises(ValueError, match='neighbor_index'):
        infer_logits(model, obs)


def test_grid_duplicate_cells():
    obs = observation()
    obs['node_features'][1] = obs['node_features'][0]
    with pytest.raises(ValueError, match='duplicate'):
        infer_logits(build_model('grid', {}), obs)


def test_device_and_precision():
    import torch
    model = build_model('node', {}, device='cpu')
    assert {str(p.device) for p in model.parameters()} == {'cpu'}
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert infer_logits(model, observation()).dtype == np.float32
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    with pytest.raises(ValueError):
        build_model('node', {}, device='meta')
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match='CUDA'):
            build_model('node', {}, device='cuda:0')


@pytest.mark.parametrize('architecture', ['graph', 'node', 'grid'])
def test_cuda_explicit_device(architecture):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; parent must run parity on admitted hardware')
    model = build_model(architecture, {}, device='cuda:0')
    assert all(p.device == torch.device('cuda:0') for p in model.parameters())
    first = infer_logits(model, observation())
    np.testing.assert_array_equal(first, infer_logits(model, observation()))


def test_geometry_and_presets():
    assert normalize_config('graph', {})['max_nodes'] == 164
    with pytest.raises(ValueError):
        build_model('unknown', {})
    for config in ({'max_nodes': 12}, {'road_length': 30}, {'num_lanes': True}):
        with pytest.raises(ValueError):
            normalize_config('graph', config)
    with pytest.raises(ValueError):
        normalize_config('grid', {'model_preset': 'grid-cnn-large'})


@pytest.mark.parametrize('weights', [None, {'tiny': np.zeros(1, np.float32)}])
@pytest.mark.parametrize('hidden,layers', [(1024, 1024), (1024, 12), (128, 807)])
def test_graph_size_rejected_before_torch(monkeypatch, weights, hidden, layers):
    import sys
    from types import SimpleNamespace
    from physica_mc import models

    def forbidden(*args, **kwargs):
        pytest.fail('oversize config reached Torch configuration or construction')

    monkeypatch.setattr(models, 'configure_torch', forbidden)
    monkeypatch.setitem(sys.modules, 'physica_mc.compat.graph', SimpleNamespace(build_graph=forbidden))
    config = dict(hidden_dim=hidden, message_layers=layers)
    with pytest.raises(ValueError, match='FP32 parameters exceed size limit'):
        normalize_config('graph', config)
    with pytest.raises(ValueError, match='FP32 parameters exceed size limit'):
        build_model('graph', config, weights)


@pytest.mark.parametrize('hidden,layers', [(128, 3), (1024, 11), (128, 806), (1, 1024)])
def test_graph_size_limit_preserves_supported_configs(hidden, layers):
    config = normalize_config('graph', dict(hidden_dim=hidden, message_layers=layers))
    assert (config['hidden_dim'], config['message_layers']) == (hidden, layers)


@pytest.mark.parametrize('hidden,layers', [(1, 1), (7, 2), (32, 4)])
def test_graph_parameter_bound_matches_constructor(hidden, layers):
    model = build_model('graph', dict(hidden_dim=hidden, message_layers=layers))
    expected = 6 * hidden**2 + 62 * hidden + 4 + layers * (5 * hidden**2 + 9 * hidden)
    assert sum(p.numel() for p in model.parameters()) == expected
