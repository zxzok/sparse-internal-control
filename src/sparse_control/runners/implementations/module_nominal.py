"""
Task-specific executed experiment implementation.
"""
import argparse
import json
import time
from sparse_control.selection.core import DriverCircuitDiscovery
from sparse_control.selection.execution_context import actual_gain_greedy_baseline, coverage_driver_baseline, random_matched_baseline, top_node_attribution_baseline
from sparse_control.runners.implementations.nominal_context import build_mcqa_gemma, resolve_device, set_determinism, summarize_result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=['Qwen/Qwen3-4B', 'mistralai/Mistral-7B-Instruct-v0.1'])
    p.add_argument('--device', default='mps')
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--seeds', required=True)
    p.add_argument('--ks', default='8', choices=['8'])
    p.add_argument('--epsilon', type=float, default=0.2)
    p.add_argument('--ridge', type=float, required=True)
    p.add_argument('--mcqa-source', required=True, choices=['mmlu_humanities', 'mmlu_stem'])
    p.add_argument('--methods', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    methods = args.methods.split(',')
    allowed = ['driver_omp', 'coverage_driver', 'top_attribution', 'random_matched', 'actual_gain_greedy']
    if methods not in [allowed, allowed[:4]]:
        p.error('Choose the reported four-method or five-method sequence')
    device = resolve_device(args.device)
    with open(args.out, 'x') as stream:
        for seed in map(int, args.seeds.split(',')):
            set_determinism(seed)
            args.seed = seed
            adapter, target, nodes, clean, corrupt = build_mcqa_gemma(args, device)
            k = 8
            disc = DriverCircuitDiscovery(adapter=adapter, target_metric=target, candidate_nodes=nodes, epsilon=args.epsilon, max_drivers=k, ridge=args.ridge, lin_tol=0.001, target_sketch_dim=None, sketch_seed=seed, fallback_mode='actual_gain', max_prune_queries=64, alpha_clip=None)
            calls = {'driver_omp': lambda: disc.run(clean, corrupt), 'coverage_driver': lambda: coverage_driver_baseline(disc, clean, corrupt, k=k, coverage_target=0.95), 'top_attribution': lambda: top_node_attribution_baseline(disc, clean, corrupt, k=k), 'random_matched': lambda: random_matched_baseline(disc, clean, corrupt, k=k, seed=seed), 'actual_gain_greedy': lambda: actual_gain_greedy_baseline(disc, clean, corrupt, k=k, topk_proposal=32)}
            for name in methods:
                started = time.time()
                result = calls[name]()
                record = summarize_result(name, seed, k, result, time.time() - started)
                if name in ['driver_omp', 'coverage_driver']:
                    record['ridge'] = args.ridge
                if name == 'coverage_driver':
                    history = result.validation_log[0].get('coverage_history') if result.validation_log else None
                    if history is not None:
                        record['coverage_history'] = history
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                print(seed, name, record['rel_l2_to_clean'], flush=True)
if __name__ == '__main__':
    main()
