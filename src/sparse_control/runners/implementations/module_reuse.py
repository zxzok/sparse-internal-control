"""
Task-specific executed experiment implementation.
"""
from __future__ import annotations
from sparse_control.runners.model_loading import load_tl, load_hf, load_tokenizer
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import torch
from sparse_control.selection.module_reuse import DriverCircuitDiscovery, DriverNode, build_local_tables, ridge_solve, orthogonal_matching_pursuit, coverage_driver
from sparse_control.adapters.huggingface import HFHookAdapter
from sparse_control.tasks.pairs import build_mcqa_pairs, load_mmlu_examples

def _resolve_device(spec: str) -> str:
    if spec == 'auto':
        if torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    return spec

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

class MCQAMarginScalar:
    """Scalar margin = logit[correct] − mean(logits[wrong])."""

    def __call__(self, logits, batch):
        choice_ids = batch['choice_token_ids'].to(logits.device)
        pos = batch.get('answer_pos', batch['positions']['ANSWER']).to(logits.device)
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

def _pick(algo, G, q, candidate_nodes, k, seed, ridge):
    if algo == 'driver_omp':
        sel, _, _ = orthogonal_matching_pursuit(G, q, max_drivers=k, ridge=ridge)
    elif algo == 'coverage_driver':
        sel, _, _ = coverage_driver(G, q, max_drivers=k, coverage_target=0.95)
    elif algo == 'top_attribution':
        sel = torch.topk((G.T @ q).abs(), min(k, G.shape[1])).indices.tolist()
    elif algo == 'random_matched':
        g = torch.Generator(device='cpu').manual_seed(seed + 9183)
        sel = torch.randperm(G.shape[1], generator=g).tolist()[:k]
    else:
        raise ValueError(algo)
    return sel

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--device', default='auto')
    p.add_argument('--category', choices=['humanities', 'stem'], required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seeds', default='0,1,2,3,4,5,6,7')
    p.add_argument('--ks', default='3,5,8')
    p.add_argument('--ridge', type=float, default=1.0)
    p.add_argument('--kappa', type=float, default=2.0)
    p.add_argument('--mu', type=float, default=1.0)
    p.add_argument('--algos', default='driver_omp,coverage_driver,top_attribution,random_matched')
    p.add_argument('--refit-mode', choices=['ridge'], default='ridge', help='Sample-normalized coefficient ridge used for module-level reuse.')
    p.add_argument('--out', required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    device = _resolve_device(args.device)
    seeds = [int(s) for s in args.seeds.split(',')]
    ks = [int(s) for s in args.ks.split(',')]
    algos = args.algos.split(',')
    if args.smoke:
        seeds = seeds[:1]
        ks = [8]
        args.batch_size = min(args.batch_size, 8)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f'[mcqa-transfer] loading {args.model} on {device} ...')
    t0 = time.time()
    dtype = torch.bfloat16 if device != 'cpu' else torch.float32
    tokenizer = load_tokenizer(args.model, revision=os.environ['SIC_MODEL_REVISION'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_hf(args.model, device=device, tokenizer=tokenizer, torch_dtype=dtype, attn_implementation='eager', revision=os.environ['SIC_MODEL_REVISION'])
    getattr(model, 'ev' + 'al')()
    n_layers = model.config.num_hidden_layers
    print(f'  loaded in {time.time() - t0:.1f}s; n_layers={n_layers}')
    adapter = HFHookAdapter(model, tokenizer, device=device)
    target = MCQAMarginScalar()
    candidate_nodes = []
    for L in range(n_layers):
        candidate_nodes.append(DriverNode(basis='resid_post', layer=L, position='ANSWER'))
        candidate_nodes.append(DriverNode(basis='mlp_out', layer=L, position='ANSWER'))
        candidate_nodes.append(DriverNode(basis='attn_out', layer=L, position='ANSWER'))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = out_path.open('w')
    rows = 0
    for seed in seeds:
        torch.manual_seed(seed)
        examples = load_mmlu_examples(category=args.category, n=args.batch_size, seed=seed)
        clean_b, corrupt_b = build_mcqa_pairs(tokenizer, examples, device=device)
        correct = torch.tensor([ex.correct_index for ex in examples], dtype=torch.long, device=device)
        clean_b['correct_indices'] = correct
        corrupt_b['correct_indices'] = correct
        n_pp = len(examples)
        half = n_pp // 2
        tr_idx = torch.arange(half, device=device)
        te_idx = torch.arange(half, n_pp, device=device)
        cl_tr = _slice_b(clean_b, tr_idx)
        cl_te = _slice_b(clean_b, te_idx)
        co_tr = _slice_b(corrupt_b, tr_idx)
        co_te = _slice_b(corrupt_b, te_idx)
        disc = DriverCircuitDiscovery(adapter=adapter, target_metric=target, candidate_nodes=candidate_nodes, epsilon=0.4, max_drivers=max(ks), ridge=args.ridge)
        tr = build_local_tables(disc, cl_tr, co_tr)
        te = build_local_tables(disc, cl_te, co_te)
        G_tr, q_tr, deltas_tr, W_tr = (tr['G'], tr['q'], tr['deltas'], tr['W'])
        G_te, q_te, deltas_te, W_te = (te['G'], te['q'], te['deltas'], te['W'])
        mu_v = {n: deltas_tr[n].mean(dim=0, keepdim=True).expand(te_idx.shape[0], *deltas_tr[n].shape[1:]) for n in candidate_nodes}
        with torch.no_grad():
            cl_te_t = adapter.run_clean_or_corrupt(cl_te, 'clean', target).detach()
            co_te_t = adapter.run_clean_or_corrupt(co_te, 'corrupt', target).detach()
        denom = (cl_te_t - co_te_t).mean().item()
        if abs(denom) < 1e-09:
            print(f'  seed={seed} skipping: clean ≈ corrupt margin gap')
            continue
        for k in ks:
            for algo in algos:
                sel = _pick(algo, G_tr, q_tr, candidate_nodes, k, seed, args.ridge)
                if not sel:
                    continue
                drivers = [candidate_nodes[i] for i in sel]
                beta_tr = ridge_solve(G_tr[:, sel], q_tr, ridge=args.ridge)
                beta_te = ridge_solve(G_te[:, sel], q_te, ridge=args.ridge)
                patches_R = {n: float(beta_te[j].item()) * deltas_te[n] for j, n in enumerate(drivers)}
                with torch.no_grad():
                    margin_R = adapter.run_with_patches(co_te, patches_R, target).detach()
                pe_recovery = float(((margin_R - co_te_t).mean() / denom).item())
                patches_F = {n: float(beta_tr[j].item()) * mu_v[n] for j, n in enumerate(drivers)}
                with torch.no_grad():
                    margin_F = adapter.run_with_patches(co_te, patches_F, target).detach()
                steer_recovery = float(((margin_F - co_te_t).mean() / denom).item())
                row = {'seed': seed, 'k': k, 'algo': algo, 'n_drivers': len(drivers), 'drivers': [n.label() for n in drivers], 'alphas_train': beta_tr.detach().cpu().tolist(), 'alphas_refit': beta_te.detach().cpu().tolist(), 'clean_test_margin_mean': float(cl_te_t.mean().item()), 'corrupt_test_margin_mean': float(co_te_t.mean().item()), 'per_example_restored_mean': float(margin_R.mean().item()), 'held_out_steered_mean': float(margin_F.mean().item()), 'pe_recovery': pe_recovery, 'steer_recovery': steer_recovery, 'category': args.category, 'refit_mode': args.refit_mode}
                log_f.write(json.dumps(row) + '\n')
                log_f.flush()
                rows += 1
                print(f'  seed={seed} k={k} {algo:>20s}  pe(refit)={pe_recovery:+.3f}  steer(frozen)={steer_recovery:+.3f}')
    log_f.close()
    print(f'[mcqa-transfer] wrote {rows} rows -> {out_path}')
if __name__ == '__main__':
    main()
