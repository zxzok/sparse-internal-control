"""
Task-specific executed experiment implementation.
"""
from __future__ import annotations
from sparse_control.runners.model_loading import load_tl, load_hf, load_tokenizer
import argparse
import json
import os
import time
from typing import Dict, List
import torch
from sparse_control.selection.core import DriverCircuitDiscovery, DriverNode, ridge_solve, orthogonal_matching_pursuit, coverage_driver
from sparse_control.adapters.transformerlens import TransformerLensAdapter, IOILogitDiff
from sparse_control.tasks.pairs import build_ioi_pairs

class FullLogits:

    def __call__(self, logits, batch):
        return logits.detach()

def _pick_drivers(algo, G, q, candidate_nodes, k, seed, ridge):
    n_cand = G.shape[1]
    if algo == 'driver_omp':
        sel, _, _ = orthogonal_matching_pursuit(G, q, max_drivers=k, ridge=ridge)
    elif algo == 'coverage_driver':
        sel, _, _ = coverage_driver(G, q, max_drivers=k, coverage_target=0.95)
    elif algo == 'top_attribution':
        sel = torch.topk((G.T @ q).abs(), min(k, n_cand)).indices.tolist()
    elif algo == 'random_matched':
        g = torch.Generator(device='cpu').manual_seed(seed + 9183)
        sel = torch.randperm(n_cand, generator=g).tolist()[:k]
    elif algo == 'actadd_L9_H9':
        target_lab = 'head_pos:L9:PEND:H9'
        sel = [i for i, n in enumerate(candidate_nodes) if n.label() == target_lab][:1]
    else:
        raise ValueError(algo)
    if not sel:
        return ([], torch.empty(0))
    alphas = ridge_solve(G[:, sel], q, ridge=ridge)
    return (sel, alphas)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gpt2')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--ridge', type=float, default=1.0)
    p.add_argument('--strengths', default='-1.0,-0.5,0,0.25,0.5,0.75,1.0,1.25,1.5,2.0')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    seeds = [int(s) for s in args.seeds.split(',')]
    strengths = [float(s) for s in args.strengths.split(',')]
    from transformer_lens import HookedTransformer
    model = load_tl(args.model, device=args.device, revision=os.environ['SIC_MODEL_REVISION'])
    tokenizer = model.tokenizer
    candidate_nodes: List[DriverNode] = []
    for layer in range(model.cfg.n_layers):
        for pos in ('END', 'IO', 'S1', 'S2'):
            for head in range(model.cfg.n_heads):
                candidate_nodes.append(DriverNode(basis='head_pos', layer=layer, position=pos, component=head))
    adapter = TransformerLensAdapter(model, device=args.device)
    target_logitdiff = IOILogitDiff()
    target_full = FullLogits()
    algos = ['driver_omp', 'coverage_driver', 'top_attribution', 'random_matched', 'actadd_L9_H9']
    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        clean, corrupt = build_ioi_pairs(tokenizer, n=args.batch_size, device=args.device, seed=seed)
        disc = DriverCircuitDiscovery(adapter=adapter, target_metric=target_logitdiff, candidate_nodes=candidate_nodes, epsilon=0.1, max_drivers=args.k, ridge=args.ridge)
        G, q, deltas = disc.estimate_local_effects(clean, corrupt)
        clean_logits = adapter.run_clean_or_corrupt(clean, 'clean', target_full)
        corrupt_logits = adapter.run_clean_or_corrupt(corrupt, 'corrupt', target_full)
        clean_targval = target_logitdiff(clean_logits, clean).detach()
        corrupt_targval = target_logitdiff(corrupt_logits, corrupt).detach()
        end_pos = clean['answer_pos'].to(args.device)
        bsz = clean['tokens'].shape[0]
        seq_len = clean_logits.shape[1]
        attn_mask = clean['attention_mask'].to(args.device).bool()
        idx_seq = torch.arange(seq_len, device=args.device).unsqueeze(0)
        is_answer = idx_seq == end_pos.unsqueeze(1)
        off_mask = attn_mask & ~is_answer
        log_clean = torch.log_softmax(clean_logits.float(), dim=-1)
        for algo in algos:
            sel, alphas = _pick_drivers(algo, G, q, candidate_nodes, args.k, seed=seed, ridge=args.ridge)
            drivers = [candidate_nodes[i] for i in sel]
            if not drivers:
                continue
            for s in strengths:
                t0 = time.time()
                if s == 0.0:
                    patched_targval = corrupt_targval
                    patched_logits = corrupt_logits
                else:
                    patches = {n: float(s * alphas[j].item()) * deltas[n] for j, n in enumerate(drivers)}
                    patched_logits = adapter.run_with_patches(corrupt, patches, target_full)
                    patched_targval = target_logitdiff(patched_logits, corrupt).detach()
                log_p = torch.log_softmax(patched_logits.float(), dim=-1)
                p_p = log_p.exp()
                kl = (p_p * (log_p - log_clean)).sum(dim=-1)
                idx = torch.arange(bsz, device=args.device)
                kl_answer = kl[idx, end_pos].mean().item()
                kl_off = (kl * off_mask.float()).sum().item() / off_mask.float().sum().clamp_min(1).item()
                target_margin = patched_targval.mean().item()
                rows.append({'seed': seed, 'algo': algo, 'strength': s, 'n_drivers': len(drivers), 'target_margin': target_margin, 'clean_target_margin': clean_targval.mean().item(), 'corrupt_target_margin': corrupt_targval.mean().item(), 'kl_answer': kl_answer, 'kl_off': kl_off, 'wallclock_s': time.time() - t0})
            print(f"seed={seed} {algo:>16s}  s=0→{rows[-len(strengths)]['target_margin']:.2f}  s=1→{[r for r in rows if r['seed'] == seed and r['algo'] == algo and (r['strength'] == 1.0)][0]['target_margin']:.2f}  clean={clean_targval.mean().item():.2f}")
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'\nWrote {args.out}')
if __name__ == '__main__':
    main()
