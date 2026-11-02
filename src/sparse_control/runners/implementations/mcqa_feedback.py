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
from sparse_control.selection.core import DriverCircuitDiscovery, DriverNode, coverage_driver, orthogonal_matching_pursuit, ridge_solve
from sparse_control.adapters.transformerlens import TransformerLensAdapter
from sparse_control.tasks.pairs import build_mcqa_pairs, load_mmlu_examples

class MCQAMarginScalar:
    """Returns scalar margin = logit[correct] − mean(logits[wrong]) per row."""

    def __call__(self, logits, batch):
        choice_ids = batch['choice_token_ids'].to(logits.device)
        if 'answer_pos' in batch:
            pos = batch['answer_pos'].to(logits.device)
        else:
            pos = batch['positions']['ANSWER'].to(logits.device)
        idx = torch.arange(logits.shape[0], device=logits.device)
        ans = logits[idx, pos, :]
        choice_logits = ans.gather(-1, choice_ids)
        correct = batch['correct_indices'].to(logits.device)
        b = torch.arange(choice_logits.shape[0], device=logits.device)
        correct_l = choice_logits[b, correct]
        mask = torch.ones_like(choice_logits, dtype=torch.bool)
        mask[b, correct] = False
        wrong_mean = choice_logits.masked_fill(~mask, 0).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1).float()
        return correct_l - wrong_mean

def _broadcast_mean(delta, batch_size):
    mean = delta.mean(dim=0)
    return mean.unsqueeze(0).expand(batch_size, *mean.shape)

def _slice_b(b, idxs):
    out = {}
    for k, v in b.items():
        if k == 'positions':
            out[k] = {pk: pv[idxs] for pk, pv in v.items()}
        elif torch.is_tensor(v) and v.shape and (v.shape[0] >= idxs.shape[0]):
            out[k] = v[idxs]
        else:
            out[k] = v
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=24, help='Total examples (train+test). Smaller because Qwen3-4B is heavy.')
    p.add_argument('--ridge', type=float, default=1.0)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mcqa-source', default='mmlu_humanities')
    p.add_argument('--targets', default='-2.0,-1.0,0.0,1.0,2.0,3.0', help='Target margins to hit')
    p.add_argument('--tol', type=float, default=0.3)
    p.add_argument('--max-iters', type=int, default=12)
    p.add_argument('--lr', type=float, default=0.15)
    p.add_argument('--algos', default='driver_omp,coverage_driver,top_attribution')
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
    category = args.mcqa_source.replace('mmlu_', '')
    examples = load_mmlu_examples(category=category, n=args.batch_size, seed=args.seed)
    print(f'  loaded {len(examples)} {args.mcqa_source} examples')
    clean_b, corrupt_b = build_mcqa_pairs(tokenizer, examples, device=args.device)
    correct_idx = torch.tensor([ex.correct_index for ex in examples], dtype=torch.long, device=args.device)
    clean_b['correct_indices'] = correct_idx
    corrupt_b['correct_indices'] = correct_idx
    target = MCQAMarginScalar()
    n_pp = len(examples)
    half = n_pp // 2
    train_idx = torch.arange(half, device=args.device)
    test_idx = torch.arange(half, n_pp, device=args.device)
    clean_train, clean_test = (_slice_b(clean_b, train_idx), _slice_b(clean_b, test_idx))
    corrupt_train, corrupt_test = (_slice_b(corrupt_b, train_idx), _slice_b(corrupt_b, test_idx))
    candidate_nodes: List[DriverNode] = []
    for L in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            candidate_nodes.append(DriverNode(basis='head_pos', layer=L, position='ANSWER', component=h))
    adapter = TransformerLensAdapter(model, device=args.device)
    print(f'  candidates: {len(candidate_nodes)}')
    torch.manual_seed(args.seed)
    disc = DriverCircuitDiscovery(adapter=adapter, target_metric=target, candidate_nodes=candidate_nodes, epsilon=0.1, max_drivers=args.k, ridge=args.ridge)
    print('Estimating local effects on TRAIN ...')
    G_tr, q_tr, deltas_tr = disc.estimate_local_effects(clean_train, corrupt_train)
    print('Estimating local effects on TEST ...')
    G_te, q_te, deltas_te = disc.estimate_local_effects(clean_test, corrupt_test)
    clean_test_t = adapter.run_clean_or_corrupt(clean_test, 'clean', target).detach()
    corrupt_test_t = adapter.run_clean_or_corrupt(corrupt_test, 'corrupt', target).detach()
    print(f'\nTest: clean margin = {clean_test_t.mean():+.3f}')
    print(f'Test: corrupt margin = {corrupt_test_t.mean():+.3f}')
    print(f'Targets: {targets}')
    rows = []
    for algo in algos:
        n_cand = G_tr.shape[1]
        if algo == 'driver_omp':
            sel, _, _ = orthogonal_matching_pursuit(G_tr, q_tr, max_drivers=args.k, ridge=args.ridge)
        elif algo == 'coverage_driver':
            sel, _, _ = coverage_driver(G_tr, q_tr, max_drivers=args.k, coverage_target=0.95)
        elif algo == 'top_attribution':
            sel = torch.topk((G_tr.T @ q_tr).abs(), args.k).indices.tolist()
        else:
            print(f'  skipping unknown algo {algo}')
            continue
        if not sel:
            continue
        drivers = [candidate_nodes[i] for i in sel]
        alphas_base = ridge_solve(G_tr[:, sel], q_tr, ridge=args.ridge)

        def eval_per_example(s: float) -> float:
            if abs(s) < 1e-09:
                return float(corrupt_test_t.mean().item())
            patches = {n: float(s * alphas_base[j].item()) * deltas_te[n] for j, n in enumerate(drivers)}
            patched = adapter.run_with_patches(corrupt_test, patches, target)
            return float(patched.mean().item())
        test_size = test_idx.shape[0]
        mu = {n: _broadcast_mean(deltas_tr[n], test_size) for n in drivers}

        def eval_held_out_steer(s: float) -> float:
            if abs(s) < 1e-09:
                return float(corrupt_test_t.mean().item())
            patches = {n: float(s * alphas_base[j].item()) * mu[n] for j, n in enumerate(drivers)}
            patched = adapter.run_with_patches(corrupt_test, patches, target)
            return float(patched.mean().item())
        for mode_name, eval_fn in [('per_example', eval_per_example), ('held_out_steer', eval_held_out_steer)]:
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
                rows.append({'algo': algo, 'mode': mode_name, 'M_star': M_star, 'k': args.k, 'clean_margin': float(clean_test_t.mean().item()), 'corrupt_margin': float(corrupt_test_t.mean().item()), 'iter_final_s': s_t, 'iter_final_M': M_final, 'iter_final_err': M_final - M_star, 'converged_at': converged, 'n_evals': len(history) + 1, 'history': history})
                print(f"{algo:>20s} {mode_name:>15s}  M*={M_star:+.2f}  s={s_t:+.2f}  M={M_final:+.2f}  err={M_final - M_star:+.2f}  {('conv@' + str(converged) if converged is not None else 'no-conv')}  {len(history) + 1} evals")
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'\nWrote {len(rows)} rows to {args.out}')
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[r['algo'], r['mode']].append(r)
    print('\n=== Convergence summary (target |err| < tol) ===')
    print(f"{'algo':>20s}  {'mode':>15s}  {'#conv':>5s}/{'#tot':>4s}  {'mean err':>9s}  {'mean s':>7s}  {'mean evals':>10s}")
    for (algo, mode), recs in sorted(agg.items()):
        nconv = sum((1 for r in recs if r['converged_at'] is not None))
        ntot = len(recs)
        m_err = sum((abs(r['iter_final_err']) for r in recs)) / max(1, ntot)
        m_s = sum((r['iter_final_s'] for r in recs)) / max(1, ntot)
        m_ev = sum((r['n_evals'] for r in recs)) / max(1, ntot)
        print(f'{algo:>20s}  {mode:>15s}  {nconv:>5d}/{ntot:>4d}  {m_err:>9.3f}  {m_s:>+7.2f}  {m_ev:>10.1f}')
if __name__ == '__main__':
    main()
