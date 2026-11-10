"""Run one final-protocol training line on an explicitly sized local Ray runtime."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

from training import checkpoints as cp
from training.configuration import configure, set_driver_threads, worker_setup

ROOT = Path(__file__).resolve().parents[1]


def sources():
    closure = json.loads((ROOT / 'reproduction/source-closure.json').read_text())
    paths = [ROOT / name for name in closure if name.startswith(('src/', 'research/'))]
    paths.extend(sorted((ROOT / 'reproduction').rglob('*.py')))
    return {p.relative_to(ROOT).as_posix(): cp.sha(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--line', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ray-cpus', required=True, type=int)
    parser.add_argument('--worker-cpus', required=True, type=float)
    parser.add_argument('--device', required=True, choices=('cpu', 'cuda'))
    parser.add_argument('--restore', type=Path, help='trusted local COMPLETE checkpoint directory')
    args = parser.parse_args()
    lines = json.loads((ROOT / 'reproduction/configs/training-lines.json').read_text())
    matches = [line for line in lines if line['line'] == args.line]
    cp.require(len(matches) == 1, 'unknown frozen training line')
    line = matches[0]
    cp.require(math.isfinite(args.worker_cpus) and args.worker_cpus > 0
               and args.ray_cpus >= 1 + 64 * args.worker_cpus,
               'local CPU reservation must fit all 64 actors and learner')
    out = args.output.resolve()
    cp.require(not out.exists(), 'use a new output directory, including for explicit restore')
    science = cp.scientific_config(line['architecture'], line['seed'])
    if line['penalty_scale'] != 1:
        science['penalty_scale'] = line['penalty_scale']
        science['env']['applied_penalty'] *= line['penalty_scale']
        science['env']['rejected_penalty'] *= line['penalty_scale']
    source_hashes = sources()
    expected = cp.identity(science, source_hashes)
    restored = cp.validate_checkpoint(args.restore.resolve(), expected) if args.restore else None
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'research/nograph'), str(ROOT / 'research')]
    import torch
    import ray
    import train_pv_graph_ppo as graph
    import train_pv_nongraph_ppo as nongraph
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env
    from snfs_traffic.rl.centralized_pv_env import CentralizedPvEnv
    from snfs_traffic.rl.graph_ppo import FactorizedGraphPPO, PvGraphTorchModel
    from snfs_traffic.rl.nongraph_ppo import PvNonGraphTorchModel

    cp.require(not ray.is_initialized(), 'requires its own local Ray runtime')
    cp.require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
    set_driver_threads(torch, 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    register_env('physica_local', lambda context: CentralizedPvEnv(context))
    ModelCatalog.register_custom_model(graph.MODEL_NAME, PvGraphTorchModel)
    ModelCatalog.register_custom_model(nongraph.MODEL_NAME, PvNonGraphTorchModel)
    owner = graph if line['architecture'] == 'graph' else nongraph
    config, raw = configure(science, 'physica_local', graph, owner, args.worker_cpus, args.device)
    out.mkdir(parents=True)
    (out / 'checkpoints').mkdir()
    cp.json_write(out / 'run_config.json', {'identity': expected, 'scientific': science,
        'sources': source_hashes, 'effective_config': cp.scalar_tree(raw),
        'runtime': {'ray': ray.__version__, 'torch': torch.__version__, 'device': args.device},
        'resume': 'functional only; sampler/environment/RNG caches are not restored'})
    algorithm = None
    try:
        ray.init(address='local', num_cpus=args.ray_cpus,
                 num_gpus=1 if args.device == 'cuda' else 0,
                 runtime_env={'worker_process_setup_hook': worker_setup,
                              'env_vars': {k: '1' for k in
                                           ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                            'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')}})
        algorithm = FactorizedGraphPPO(config=config)
        if restored:
            manifest, payload = restored
            algorithm.restore(str(payload))
            cp.assert_resume(cp.algorithm_snapshot(algorithm), manifest['snapshot'])
        parameters = list(algorithm.get_policy().model.parameters())
        cp.require(sum(p.numel() for p in parameters) == science['parameters']
                   and all(p.dtype == torch.float32 and p.device.type == args.device for p in parameters),
                   'learner parameter count/dtype/device mismatch')
        iteration = int(algorithm.iteration)
        while iteration < line['iterations']:
            cp.require(sources() == source_hashes, 'source changed during training')
            result = algorithm.train()
            iteration = cp.check_steps(result, iteration)
            stats = graph._nested(result, ('learners', 'default_policy', 'learner_stats'),
                                  ('info', 'learner', 'default_policy', 'learner_stats'))
            cp.require(isinstance(stats, dict) and 'total_loss' in stats
                       and math.isfinite(float(stats['total_loss'])), 'missing/nonfinite total loss')
            cp.require(all(bool(torch.isfinite(p).all()) for p in parameters), 'nonfinite model')
            for optimizer in algorithm.get_policy()._optimizers:
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            cp.require(bool(torch.isfinite(value).all()), 'nonfinite optimizer')
                            cp.require(key == 'step' or value.device.type == args.device,
                                       'optimizer device mismatch')
            cp.require(sources() == source_hashes, 'source changed during iteration')
            snapshot = cp.algorithm_snapshot(algorithm)
            cp.require(snapshot['iteration'] == iteration and snapshot['sampled'] == snapshot['trained']
                       == iteration * cp.BATCH, 'snapshot counters mismatch')
            cp.publish_checkpoint(algorithm, out / 'checkpoints', expected, snapshot,
                                  graph._save_unique_checkpoint)
            print(json.dumps({'iteration': iteration, 'sampled': snapshot['sampled']}), flush=True)
    finally:
        try:
            if algorithm is not None:
                algorithm.stop()
        finally:
            ray.shutdown()


if __name__ == '__main__':
    main()
