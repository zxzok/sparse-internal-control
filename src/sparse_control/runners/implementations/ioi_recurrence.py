"""
Task-specific executed experiment implementation.
"""
import argparse
import json
import time
from sparse_control.selection.core import DriverCircuitDiscovery
from sparse_control.runners.implementations.nominal_context import build_ioi_gpt2, resolve_device, set_determinism, summarize_result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='gpt2')
    p.add_argument('--device', default='mps')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seeds', default='0,1,2,3,4')
    p.add_argument('--ks', default='3,5,8,12,16')
    p.add_argument('--ridge', type=float, default=1.0)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = resolve_device(args.device)
    with open(args.out, 'x') as f:
        for seed in map(int, args.seeds.split(',')):
            set_determinism(seed)
            args.seed = seed
            adapter, target, nodes, clean, corrupt = build_ioi_gpt2(args, device)
            for k in map(int, args.ks.split(',')):
                disc = DriverCircuitDiscovery(adapter=adapter, target_metric=target, candidate_nodes=nodes, epsilon=0.1, max_drivers=k, ridge=args.ridge, lin_tol=0.001, target_sketch_dim=None, sketch_seed=seed, fallback_mode='actual_gain', max_prune_queries=64)
                start = time.time()
                result = disc.run(clean, corrupt)
                record = summarize_result('driver_omp', seed, k, result, time.time() - start)
                record['ridge'] = args.ridge
                f.write(json.dumps(record) + '\n')
                f.flush()
                print('seed', seed, 'k', k, flush=True)
if __name__ == '__main__':
    main()
