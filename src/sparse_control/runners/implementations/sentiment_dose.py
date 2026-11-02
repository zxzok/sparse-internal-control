"""
Task-specific executed experiment implementation.
"""
from __future__ import annotations
from sparse_control.runners.model_loading import load_tl, load_hf, load_tokenizer
import argparse
import json
import os
from typing import Any, Dict, List, Mapping
import torch
from sparse_control.selection.core import DriverNode, ridge_solve, orthogonal_matching_pursuit, coverage_driver
from sparse_control.adapters.transformerlens import TransformerLensAdapter
from sparse_control.tasks.sentiment_unpaired import SentimentLogitDiff, _build_batch as _sent_build, _slice_batch as _sent_slice, estimate_unpaired_local_effects, _POS_PROMPTS, _NEG_PROMPTS, _POS_WORDS, _NEG_WORDS
SCALES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

def run_sentiment(args, rows):
    from transformer_lens import HookedTransformer
    model = load_tl('gpt2', device=args.device, revision=os.environ['SIC_MODEL_REVISION'])
    tokenizer = model.tokenizer
    pos_ids = torch.tensor([tokenizer.encode(' ' + w, add_special_tokens=False)[0] for w in _POS_WORDS], device=args.device)
    neg_ids = torch.tensor([tokenizer.encode(' ' + w, add_special_tokens=False)[0] for w in _NEG_WORDS], device=args.device)
    target = SentimentLogitDiff(pos_ids, neg_ids)
    candidate_nodes = [DriverNode(basis='head_pos', layer=L, position='ANSWER', component=h) for L in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    adapter = TransformerLensAdapter(model, device=args.device)
    actadd_layer = model.cfg.n_layers // 2 + 1
    actadd_node_resid = DriverNode(basis='resid_pos', layer=actadd_layer, position='ANSWER', name=f'actadd_resid_L{actadd_layer}')
    import random as _rnd
    for seed in args.seeds:
        torch.manual_seed(seed)
        rng = _rnd.Random(seed)
        idx_perm = list(range(len(_POS_PROMPTS)))
        rng.shuffle(idx_perm)
        pos_p = [_POS_PROMPTS[i] for i in idx_perm]
        neg_p = [_NEG_PROMPTS[i] for i in idx_perm]
        pos_batch = _sent_build(tokenizer, pos_p, args.device)
        neg_batch = _sent_build(tokenizer, neg_p, args.device)
        n = pos_batch['tokens'].shape[0]
        half = n // 2
        train_idx = torch.arange(half, device=args.device)
        test_idx = torch.arange(half, n, device=args.device)
        pos_train = _sent_slice(pos_batch, train_idx)
        pos_test = _sent_slice(pos_batch, test_idx)
        neg_train = _sent_slice(neg_batch, train_idx)
        neg_test = _sent_slice(neg_batch, test_idx)
        G_tr, q_tr, mean_dirs_tr = estimate_unpaired_local_effects(adapter, target, candidate_nodes, pos_train, neg_train)
        pos_resid = adapter.cache_activations(pos_train, [actadd_node_resid])[actadd_node_resid]
        neg_resid = adapter.cache_activations(neg_train, [actadd_node_resid])[actadd_node_resid]
        actadd_mu_resid = (pos_resid.mean(dim=0) - neg_resid.mean(dim=0)).detach()
        pos_test_t = adapter.run_clean_or_corrupt(pos_test, 'clean', target).detach()
        neg_test_t = adapter.run_clean_or_corrupt(neg_test, 'corrupt', target).detach()
        with torch.no_grad():
            neg_test_logits_clean = model(neg_test['tokens']).detach()
            idx_b = torch.arange(neg_test['tokens'].shape[0], device=args.device)
            ans_idx = neg_test['answer_pos']
            log_p_clean = torch.log_softmax(neg_test_logits_clean[idx_b, ans_idx, :], dim=-1)
        denom = max((pos_test_t.mean() - neg_test_t.mean()).item(), 1e-06)

        def _patch_apply(drivers, alphas, mean_dirs, scale):
            test_size = test_idx.shape[0]
            patches = {n_v: scale * float(alphas[j].item()) * mean_dirs[n_v].unsqueeze(0).expand(test_size, *mean_dirs[n_v].shape) for j, n_v in enumerate(drivers)}
            patches_by_hook: Dict[str, Any] = {}
            for node, delta in patches.items():
                patches_by_hook.setdefault(adapter.hook_name(node), []).append((node, delta))

            def make_hook(name):

                def patch_hook(act, hook):
                    out = act
                    for node, delta in patches_by_hook[name]:
                        out = adapter._scatter_into_full(out, node, delta, alpha=1.0, batch=neg_test)
                    return out
                return patch_hook
            hooks = [(name, make_hook(name)) for name in patches_by_hook]
            with torch.no_grad():
                logits_p = model.run_with_hooks(neg_test['tokens'], fwd_hooks=hooks)
                tgt = target(logits_p, neg_test).detach()
                log_p_patched = torch.log_softmax(logits_p[idx_b, ans_idx, :], dim=-1)
                kl = (log_p_clean.exp() * (log_p_clean - log_p_patched)).sum(-1).mean().item()
            return (tgt, kl)
        kk = 8
        n_cand = G_tr.shape[1]
        algos: Dict[str, Dict[str, Any]] = {}
        sel_omp, _, _ = orthogonal_matching_pursuit(G_tr, q_tr, max_drivers=kk, ridge=args.ridge)
        sel_cov, _, _ = coverage_driver(G_tr, q_tr, max_drivers=kk, coverage_target=0.95)
        sel_top = torch.topk((G_tr.T @ q_tr).abs(), min(kk, n_cand)).indices.tolist()
        rng_t = torch.Generator(device='cpu').manual_seed(99 + seed)
        sel_rnd = torch.randperm(n_cand, generator=rng_t).tolist()[:kk]
        for name, sel in [('driver_omp', sel_omp), ('coverage_driver', sel_cov), ('top_attribution', sel_top), ('random_matched', sel_rnd)]:
            if not sel:
                continue
            drivers = [candidate_nodes[i] for i in sel]
            alphas = ridge_solve(G_tr[:, sel], q_tr, ridge=args.ridge)
            algos[name] = (drivers, alphas, mean_dirs_tr)
        jac_resid = adapter.jacobian_wrt_nodes(neg_train, [actadd_node_resid], target, target_sketch=None)[actadd_node_resid]
        g_actadd = (jac_resid * actadd_mu_resid.unsqueeze(0)).sum(dim=-1)
        alpha_a = float((q_tr * g_actadd).sum() / (g_actadd.pow(2).sum() + args.ridge))
        algos[f'actadd_resid_L{actadd_layer}'] = ([actadd_node_resid], torch.tensor([alpha_a], device=q_tr.device, dtype=q_tr.dtype), {actadd_node_resid: actadd_mu_resid})
        for scale in SCALES:
            for algo, (drivers, alphas, mean_dirs) in algos.items():
                tgt, kl = _patch_apply(drivers, alphas, mean_dirs, scale)
                rec = (tgt.mean().item() - neg_test_t.mean().item()) / denom
                rows.append({'mode': 'sentiment', 'seed': seed, 'k': kk, 'algo': algo, 'scale': scale, 'steer_recovery': rec, 'off_target_kl_mean': kl, 'neg_test_target': neg_test_t.mean().item(), 'pos_test_target': pos_test_t.mean().item()})
            print(f"  sentiment seed={seed} scale={scale:.2f}: top={[r['steer_recovery'] for r in rows[-len(algos):]]}")

def main():
    p = argparse.ArgumentParser(description='Unpaired template Sentiment dose sweep')
    p.add_argument('--device', default='auto')
    p.add_argument('--ridge', type=float, default=1.0)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seeds = [int(x) for x in args.seeds.split(',')]
    rows = []
    run_sentiment(args, rows)
    with open(args.out, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
if __name__ == '__main__':
    main()
