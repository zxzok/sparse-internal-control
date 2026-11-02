"""
Task-specific executed experiment implementation.
"""
from __future__ import annotations
from sparse_control.runners.model_loading import load_tl, load_hf, load_tokenizer
import argparse
import json
import os
from typing import Dict, List, Tuple
import torch
from sparse_control.selection.core import DriverCircuitDiscovery, DriverNode, ridge_solve, orthogonal_matching_pursuit, coverage_driver
from sparse_control.adapters.transformerlens import TransformerLensAdapter, IOILogitDiff
from sparse_control.tasks.pairs import build_ioi_pairs

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gpt2')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--ridge', type=float, default=1.0)
    p.add_argument('--targets', default='-3.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0', help='Target IO−S margins to track.')
    p.add_argument('--tol', type=float, default=0.3)
    p.add_argument('--max-iters', type=int, default=12)
    p.add_argument('--lr', type=float, default=0.15)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    seeds = [int(s) for s in args.seeds.split(',')]
    targets = [float(s) for s in args.targets.split(',')]
    from transformer_lens import HookedTransformer
    model = load_tl(args.model, device=args.device, revision=os.environ['SIC_MODEL_REVISION'])
    tokenizer = model.tokenizer
    candidate_nodes: List[DriverNode] = []
    for L in range(model.cfg.n_layers):
        for pos in ('END', 'IO', 'S1', 'S2'):
            for h in range(model.cfg.n_heads):
                candidate_nodes.append(DriverNode(basis='head_pos', layer=L, position=pos, component=h))
    adapter = TransformerLensAdapter(model, device=args.device)
    target = IOILogitDiff()
    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        clean, corrupt = build_ioi_pairs(tokenizer, n=args.batch_size, device=args.device, seed=seed)
        disc = DriverCircuitDiscovery(adapter=adapter, target_metric=target, candidate_nodes=candidate_nodes, epsilon=0.1, max_drivers=args.k, ridge=args.ridge)
        G, q, deltas = disc.estimate_local_effects(clean, corrupt)
        sel, _, _ = orthogonal_matching_pursuit(G, q, max_drivers=args.k, ridge=args.ridge)
        drivers = [candidate_nodes[i] for i in sel]
        alphas_base = ridge_solve(G[:, sel], q, ridge=args.ridge)
        clean_t = adapter.run_clean_or_corrupt(clean, 'clean', target).detach()
        corrupt_t = adapter.run_clean_or_corrupt(corrupt, 'corrupt', target).detach()

        def eval_at_strength(s: float) -> float:
            if s == 0.0:
                return float(corrupt_t.mean().item())
            patches = {n: float(s * alphas_base[j].item()) * deltas[n] for j, n in enumerate(drivers)}
            patched = adapter.run_with_patches(corrupt, patches, target)
            return float(patched.mean().item())
        for M_star in targets:
            grid = [-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
            best_s = None
            best_err = float('inf')
            n_evals_grid = 0
            grid_history = []
            for s in grid:
                M = eval_at_strength(s)
                n_evals_grid += 1
                grid_history.append({'s': s, 'M': M})
                err = abs(M - M_star)
                if err < best_err:
                    best_err = err
                    best_s = s
            ls_M_final = eval_at_strength(best_s)
            n_evals_grid += 1
            s_t = 0.0
            iter_history = []
            converged_iter = None
            for t in range(args.max_iters):
                M_t = eval_at_strength(s_t)
                err = M_t - M_star
                iter_history.append({'t': t, 's': s_t, 'M': M_t, 'err': err})
                if abs(err) < args.tol:
                    converged_iter = t
                    break
                s_t = s_t - args.lr * err / max(abs(args.lr * err), 0.5)
                s_t = float(max(min(s_t + (M_star - M_t) * args.lr, s_t + 0.5), s_t - 0.5))
                s_t = iter_history[-1]['s'] + args.lr * (M_star - M_t)
            iter_M_final = eval_at_strength(s_t) if iter_history else float('nan')
            rows.append({'seed': seed, 'M_star': M_star, 'clean_margin': float(clean_t.mean().item()), 'corrupt_margin': float(corrupt_t.mean().item()), 'line_search_best_s': best_s, 'line_search_M_final': ls_M_final, 'line_search_err': ls_M_final - M_star, 'line_search_n_evals': n_evals_grid, 'iter_final_s': s_t, 'iter_M_final': iter_M_final, 'iter_err': iter_M_final - M_star, 'iter_converged_at': converged_iter, 'iter_n_evals': len(iter_history) + 1, 'iter_history': iter_history})
            print(f"seed={seed} M*={M_star:>+5.1f}  LS s={best_s:+.2f} → M={ls_M_final:+.2f} (err {ls_M_final - M_star:+.2f}, {n_evals_grid} evals)  ITER s={s_t:+.2f} → M={iter_M_final:+.2f} (err {iter_M_final - M_star:+.2f}, {('conv@' + str(converged_iter) if converged_iter is not None else 'no-conv')}, {len(iter_history) + 1} evals)")
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'\nWrote {args.out}')
if __name__ == '__main__':
    main()
