"""
Task-specific executed experiment implementation.
"""
from __future__ import annotations
from sparse_control.runners.model_loading import load_tl, load_hf, load_tokenizer
import argparse
import json
import os
import random as _rnd
from typing import Dict, List
import torch
from sparse_control.selection.core import DriverCircuitDiscovery, DriverNode, ridge_solve, coverage_driver, orthogonal_matching_pursuit
from sparse_control.adapters.transformerlens import TransformerLensAdapter
from sparse_control.tasks.refusal import _HARMFUL_PROMPTS, _HARMLESS_PROMPTS, _REFUSAL_WORDS, _COMPLIANCE_WORDS, RefusalLogitDiff, _build_batch, _broadcast_mean

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--device', default='auto')
    p.add_argument('--ridge', type=float, default=1.0)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--targets', default='-1.4,-1.0,-0.6,-0.3,0.0', help='Target refusal margins (M*) to track')
    p.add_argument('--tol', type=float, default=0.05)
    p.add_argument('--max-iters', type=int, default=15)
    p.add_argument('--lr', type=float, default=0.5)
    p.add_argument('--algos', default='top_attribution,coverage_driver,driver_omp')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    targets = [float(s) for s in args.targets.split(',')]
    algos = args.algos.split(',')
    from transformer_lens import HookedTransformer
    print(f'Loading {args.model}...')
    model = load_tl(args.model, device=args.device, revision=os.environ['SIC_MODEL_REVISION'])
    tokenizer = model.tokenizer
    print(f'  n_layers={model.cfg.n_layers}  n_heads={model.cfg.n_heads}')

    def _tok_first(words):
        ids = []
        for w in words:
            t = tokenizer.encode(' ' + w, add_special_tokens=False)
            if t:
                ids.append(t[0])
        return torch.tensor(ids, device=args.device, dtype=torch.long)
    refusal_ids = _tok_first(_REFUSAL_WORDS)
    compliance_ids = _tok_first(_COMPLIANCE_WORDS)
    target = RefusalLogitDiff(refusal_ids, compliance_ids)
    candidate_nodes: List[DriverNode] = []
    for L in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            candidate_nodes.append(DriverNode(basis='head_pos', layer=L, position='ANSWER', component=h))
    adapter = TransformerLensAdapter(model, device=args.device)
    torch.manual_seed(args.seed)
    rng = _rnd.Random(args.seed)
    idx_perm = list(range(len(_HARMFUL_PROMPTS)))
    rng.shuffle(idx_perm)
    harmful_p = [_HARMFUL_PROMPTS[i] for i in idx_perm]
    harmless_p = [_HARMLESS_PROMPTS[i] for i in idx_perm]
    harmful_batch = _build_batch(tokenizer, harmful_p, args.device)
    harmless_batch = _build_batch(tokenizer, harmless_p, args.device)
    n_pp = len(_HARMFUL_PROMPTS)
    half = n_pp // 2
    train_idx = torch.arange(half, device=args.device)
    test_idx = torch.arange(half, n_pp, device=args.device)

    def slice_b(b, idxs):
        return {'tokens': b['tokens'][idxs], 'attention_mask': b['attention_mask'][idxs], 'answer_pos': b['answer_pos'][idxs], 'positions': {'ANSWER': b['positions']['ANSWER'][idxs]}}
    harmful_train = slice_b(harmful_batch, train_idx)
    harmful_test = slice_b(harmful_batch, test_idx)
    harmless_train = slice_b(harmless_batch, train_idx)
    harmless_test = slice_b(harmless_batch, test_idx)
    disc = DriverCircuitDiscovery(adapter=adapter, target_metric=target, candidate_nodes=candidate_nodes, epsilon=0.1, max_drivers=args.k, ridge=args.ridge)
    G_tr, q_tr, deltas_tr = disc.estimate_local_effects(harmful_train, harmless_train)
    G_te, q_te, deltas_te = disc.estimate_local_effects(harmful_test, harmless_test)
    harmful_test_t = adapter.run_clean_or_corrupt(harmful_test, 'clean', target).detach()
    harmless_test_t = adapter.run_clean_or_corrupt(harmless_test, 'corrupt', target).detach()
    print(f'\nTest: harmful_test refusal margin = {harmful_test_t.mean():+.3f}')
    print(f'Test: harmless_test refusal margin = {harmless_test_t.mean():+.3f}')
    print(f'Targets to hit: {targets}')
    rows = []
    for algo in algos:
        if algo == 'top_attribution':
            sel = torch.topk((G_tr.T @ q_tr).abs(), args.k).indices.tolist()
        elif algo == 'coverage_driver':
            sel, _, _ = coverage_driver(G_tr, q_tr, max_drivers=args.k, coverage_target=0.95)
        elif algo == 'driver_omp':
            sel, _, _ = orthogonal_matching_pursuit(G_tr, q_tr, max_drivers=args.k, ridge=args.ridge)
        else:
            print(f'  skipping unknown algo {algo}')
            continue
        if not sel:
            continue
        drivers = [candidate_nodes[i] for i in sel]
        alphas_base = ridge_solve(G_tr[:, sel], q_tr, ridge=args.ridge)

        def eval_at_strength(s: float) -> float:
            if abs(s) < 1e-09:
                return float(harmless_test_t.mean().item())
            patches = {n: float(s * alphas_base[j].item()) * deltas_te[n] for j, n in enumerate(drivers)}
            patched = adapter.run_with_patches(harmless_test, patches, target)
            return float(patched.mean().item())
        test_size = test_idx.shape[0]
        mu = {n: _broadcast_mean(deltas_tr[n], test_size) for n in drivers}

        def eval_at_strength_steer(s: float) -> float:
            if abs(s) < 1e-09:
                return float(harmless_test_t.mean().item())
            patches = {n: float(s * alphas_base[j].item()) * mu[n] for j, n in enumerate(drivers)}
            patched = adapter.run_with_patches(harmless_test, patches, target)
            return float(patched.mean().item())
        for mode_name, eval_fn in [('per_example', eval_at_strength), ('held_out_steer', eval_at_strength_steer)]:
            for M_star in targets:
                s_t = 0.0
                history = []
                converged = None
                for t in range(args.max_iters):
                    M_t = eval_fn(s_t)
                    err = M_t - M_star
                    history.append({'t': t, 's': s_t, 'M': M_t, 'err': err})
                    if abs(err) < args.tol:
                        converged = t
                        break
                    s_t = s_t + args.lr * (M_star - M_t)
                M_final = eval_fn(s_t) if history else float('nan')
                rows.append({'algo': algo, 'mode': mode_name, 'M_star': M_star, 'k': args.k, 'harmful_margin': float(harmful_test_t.mean().item()), 'harmless_margin': float(harmless_test_t.mean().item()), 'iter_final_s': s_t, 'iter_final_M': M_final, 'iter_final_err': M_final - M_star, 'converged_at': converged, 'n_evals': len(history) + 1, 'history': history})
                print(f"{algo:>20s} {mode_name:>15s}  M*={M_star:+.2f}  s={s_t:+.2f}  M={M_final:+.2f}  err={M_final - M_star:+.2f}  {('conv@' + str(converged) if converged is not None else 'no-conv')}  {len(history) + 1} evals")
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'\nWrote {args.out}')
if __name__ == '__main__':
    main()
