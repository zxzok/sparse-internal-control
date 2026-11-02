"""
Task-specific executed experiment implementation.
"""
from __future__ import annotations
from sparse_control.runners.model_loading import load_tl, load_hf, load_tokenizer
import argparse
import json
import os
import random as pyrandom
from collections import Counter
from typing import Dict, List
import torch
from sparse_control.selection.core import DriverCircuitDiscovery, DriverNode, ridge_solve
from sparse_control.adapters.transformerlens import TransformerLensAdapter, IOILogitDiff
from sparse_control.tasks.pairs import build_ioi_pairs

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gpt2')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ridge', type=float, default=1.0)
    p.add_argument('--ioi-jsonl', default='outputs/ioi_10seed_full.jsonl', help='Source for consensus core/backup')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
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
    clean, corrupt = build_ioi_pairs(tokenizer, n=args.batch_size, device=args.device, seed=args.seed)
    disc = DriverCircuitDiscovery(adapter=adapter, target_metric=target, candidate_nodes=candidate_nodes, epsilon=0.1, max_drivers=12, ridge=args.ridge)
    G, q, deltas = disc.estimate_local_effects(clean, corrupt)
    clean_target = adapter.run_clean_or_corrupt(clean, 'clean', target)
    corrupt_target = adapter.run_clean_or_corrupt(corrupt, 'corrupt', target)
    bsz = clean['tokens'].shape[0]
    runs = [json.loads(l) for l in open(args.ioi_jsonl) if json.loads(l).get('record_type') != 'consensus' and json.loads(l).get('method') == 'driver_omp']
    n_runs = len(runs)
    freq = Counter()
    for r in runs:
        for n in set(r['drivers']):
            freq[n] += 1
    sorted_by_freq = freq.most_common()
    core_labels = [n for n, c in sorted_by_freq if c / n_runs >= 0.6]
    backup_labels = [n for n, c in sorted_by_freq if 0 < c / n_runs < 0.6]
    print(f'Loaded {n_runs} OMP runs')
    print(f'core (γ=0.6, n={len(core_labels)}): {core_labels}')
    print(f'top backups (in order): {backup_labels[:6]}')
    label_to_node = {n.label(): n for n in candidate_nodes}
    label_to_idx = {n.label(): i for i, n in enumerate(candidate_nodes)}

    def per_prompt_rel_l2(labels):
        if not labels:
            patched = corrupt_target
        else:
            indices = [label_to_idx[lab] for lab in labels if lab in label_to_idx]
            G_sub = G[:, indices]
            u = ridge_solve(G_sub, q, ridge=args.ridge)
            patches = {}
            for lab, alpha in zip(labels, u.tolist()):
                node = label_to_node[lab]
                patches[node] = float(alpha) * deltas[node]
            patched = adapter.run_with_patches(corrupt, patches, target)
        d = (patched - clean_target).detach().cpu()
        clean_abs = clean_target.detach().cpu().abs().clamp_min(1e-06)
        return ((d.abs() / clean_abs).tolist(), patched.detach().cpu().tolist())
    groups = []
    groups.append(('EMPTY', []))
    groups.append(('CORE', core_labels))
    for m_b in range(1, min(len(backup_labels), 6) + 1):
        groups.append((f'CORE+top{m_b}', core_labels + backup_labels[:m_b]))
    rng = pyrandom.Random(args.seed + 999)
    random_labels = [n.label() for n in rng.sample(candidate_nodes, 6)]
    groups.append(('RANDOM_6', random_labels))
    heatmap = {}
    for gname, labels in groups:
        rels, patched_t = per_prompt_rel_l2(labels)
        heatmap[gname] = {'rel_per_prompt': rels, 'patched_target': patched_t, 'drivers': labels, 'size': len(labels)}
    rels_core = heatmap['CORE']['rel_per_prompt']
    rels_full = heatmap[f'CORE+top6']['rel_per_prompt'] if 'CORE+top6' in heatmap else heatmap[list(heatmap.keys())[-2]]['rel_per_prompt']
    rescue_idx = [i for i in range(bsz) if rels_core[i] > 0.3 and rels_full[i] < 0.3]
    print(f'\n{len(rescue_idx)}/{bsz} prompts where core fails (>0.3) but core+backup6 succeeds (<0.3) — these are "backup-rescued" prompts')
    out = {'n_runs': n_runs, 'core': core_labels, 'backups_top6': backup_labels[:6], 'n_prompts': bsz, 'rescue_indices': rescue_idx, 'groups': heatmap}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {args.out}')
    print('\n=== Per-group mean rel_l2 (across all prompts) ===')
    print(f"{'group':>14s}  {'size':>4s}  {'mean':>8s}  {'max':>8s}  {'frac>0.3':>9s}")
    for gname, _ in groups:
        rels = heatmap[gname]['rel_per_prompt']
        m = sum(rels) / len(rels)
        mx = max(rels)
        frac = sum((1 for r in rels if r > 0.3)) / len(rels)
        print(f"{gname:>14s}  {heatmap[gname]['size']:>4d}  {m:>8.3f}  {mx:>8.3f}  {frac:>9.2f}")
if __name__ == '__main__':
    main()
