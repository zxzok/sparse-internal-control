"""
Task-specific executed experiment implementation.
"""
from __future__ import annotations
from sparse_control.runners.model_loading import load_tl, load_hf, load_tokenizer
import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple
import torch
from torch import Tensor
from sparse_control.selection.core import DriverCircuitDiscovery, DriverNode, coverage_driver, orthogonal_matching_pursuit, ridge_solve
from sparse_control.adapters.transformerlens import IOILogitDiff, TransformerLensAdapter
from sparse_control.tasks.pairs import build_ioi_pairs
_ID_NAMES = ['John', 'Mary', 'Alice', 'Bob', 'Tom', 'Lisa', 'Mark', 'Kate', 'Paul', 'Jane', 'Mike', 'Sarah', 'David', 'Emma', 'James', 'Anna']
_OOD_NAMES = ['Aaron', 'Beatrice', 'Calvin', 'Diane', 'Edgar', 'Fiona', 'Gerald', 'Helena', 'Ivan', 'Julia', 'Keith', 'Laura', 'Martin', 'Nora', 'Oscar', 'Penelope']
_OOD_PLACES = ['museum', 'diner', 'stadium', 'cafe', 'garage', 'harbor']
_OOD_OBJECTS = ['watch', 'scarf', 'candle', 'letter', 'key', 'lamp']
_ID_TEMPLATES = ['When {A} and {B} went to the {place}, {C} gave a {obj} to', 'After {A} and {B} went to the {place}, {C} gave a {obj} to', 'While {A} and {B} were at the {place}, {C} gave a {obj} to', 'When {A} and {B} went to the {place}, {C} decided to give a {obj} to']
_OOD_TEMPLATES = ['Once {A} and {B} arrived at the {place}, {C} chose to hand the {obj} to', 'Whenever {A} and {B} stopped by the {place}, {C} ended up offering the {obj} to', 'Following the visit by {A} and {B} to the {place}, {C} reached out and passed the {obj} to', 'Soon after {A} and {B} reached the {place}, {C} elected to bring the {obj} over to']
_OOD_LONG_TEMPLATES = ['When {A} and {B} went to the {place} on a quiet afternoon together, {C} thought about it for a moment and finally gave a {obj} to', 'After {A} and {B} went to the {place} despite the rain that started early, {C} smiled, paused, and then gave a {obj} to', 'While {A} and {B} were at the {place} chatting about the weekend, {C} carefully wrapped up and handed a {obj} to', 'When {A} and {B} went to the {place} with friends from school, {C} took a deep breath before deciding to give a {obj} to']

@dataclass
class _BuilderConfig:
    templates: List[str]
    names: List[str]
    places: List[str]
    objects: List[str]
_VARIANT_CONFIGS: Dict[str, _BuilderConfig] = {'id': _BuilderConfig(_ID_TEMPLATES, _ID_NAMES, ['store', 'park', 'school', 'market', 'library', 'office'], ['ring', 'book', 'ball', 'drink', 'bottle', 'flower']), 'ood_names': _BuilderConfig(_ID_TEMPLATES, _OOD_NAMES, _OOD_PLACES, _OOD_OBJECTS), 'ood_templates': _BuilderConfig(_OOD_TEMPLATES, _ID_NAMES, ['store', 'park', 'school', 'market', 'library', 'office'], ['ring', 'book', 'ball', 'drink', 'bottle', 'flower']), 'ood_long': _BuilderConfig(_OOD_LONG_TEMPLATES, _ID_NAMES, ['store', 'park', 'school', 'market', 'library', 'office'], ['ring', 'book', 'ball', 'drink', 'bottle', 'flower']), 'ood_combined': _BuilderConfig(_OOD_TEMPLATES, _OOD_NAMES, _OOD_PLACES, _OOD_OBJECTS)}

def _leading_space_token_id(tokenizer: Any, word: str) -> int:
    ids = tokenizer.encode(' ' + word, add_special_tokens=False)
    return int(ids[0])

def _build_variant(tokenizer: Any, cfg: _BuilderConfig, n: int, device: str, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build (clean, corrupt) IOI pair for one variant config — mirrors
    ``build_ioi_pairs`` but parameterised over ``cfg``.
    """
    rng = random.Random(seed)
    clean_texts: List[str] = []
    corrupt_texts: List[str] = []
    io_names: List[str] = []
    s_names: List[str] = []
    for _ in range(n):
        template = rng.choice(cfg.templates)
        a, b = rng.sample(cfg.names, 2)
        place = rng.choice(cfg.places)
        obj = rng.choice(cfg.objects)
        clean_texts.append(template.format(A=a, B=b, place=place, C=a, obj=obj))
        corrupt_texts.append(template.format(A=b, B=a, place=place, C=b, obj=obj))
        io_names.append(b)
        s_names.append(a)
    clean_enc = tokenizer(clean_texts, return_tensors='pt', padding=True)
    corrupt_enc = tokenizer(corrupt_texts, return_tensors='pt', padding=True)
    seq_len = min(clean_enc['input_ids'].shape[1], corrupt_enc['input_ids'].shape[1])

    def _truncate(enc):
        return {k: v[:, :seq_len].to(device) for k, v in enc.items()}
    clean_tok = _truncate(clean_enc)
    corrupt_tok = _truncate(corrupt_enc)

    def _end(enc):
        return (enc['attention_mask'].sum(dim=-1) - 1).to(torch.long)
    clean_end = _end(clean_tok)
    corrupt_end = _end(corrupt_tok)
    io_ids = torch.tensor([_leading_space_token_id(tokenizer, w) for w in io_names], dtype=torch.long, device=device)
    s_ids = torch.tensor([_leading_space_token_id(tokenizer, w) for w in s_names], dtype=torch.long, device=device)

    def _find_pos(tok_row: Tensor, name_id: int, occurrence: int) -> int:
        idx = (tok_row == name_id).nonzero(as_tuple=False).flatten().tolist()
        return idx[occurrence - 1] if len(idx) >= occurrence else -1
    s1_clean, s2_clean, io_pos_clean = ([], [], [])
    s1_corr, s2_corr, io_pos_corr = ([], [], [])
    for i in range(n):
        sid = int(s_ids[i].item())
        iid = int(io_ids[i].item())
        s1_clean.append(_find_pos(clean_tok['input_ids'][i], sid, 1))
        s2_clean.append(_find_pos(clean_tok['input_ids'][i], sid, 2))
        io_pos_clean.append(_find_pos(clean_tok['input_ids'][i], iid, 1))
        s1_corr.append(_find_pos(corrupt_tok['input_ids'][i], iid, 1))
        s2_corr.append(_find_pos(corrupt_tok['input_ids'][i], iid, 2))
        io_pos_corr.append(_find_pos(corrupt_tok['input_ids'][i], sid, 1))

    def _pos_dict(end, s1, s2, io):
        return {'END': end, 'S1': torch.tensor(s1, dtype=torch.long, device=device), 'S2': torch.tensor(s2, dtype=torch.long, device=device), 'IO': torch.tensor(io, dtype=torch.long, device=device)}
    clean = {'tokens': clean_tok['input_ids'], 'attention_mask': clean_tok['attention_mask'], 'io_token_ids': io_ids, 's_token_ids': s_ids, 'answer_pos': clean_end, 'positions': _pos_dict(clean_end, s1_clean, s2_clean, io_pos_clean)}
    corrupt = {'tokens': corrupt_tok['input_ids'], 'attention_mask': corrupt_tok['attention_mask'], 'io_token_ids': io_ids, 's_token_ids': s_ids, 'answer_pos': corrupt_end, 'positions': _pos_dict(corrupt_end, s1_corr, s2_corr, io_pos_corr)}
    return (clean, corrupt)

def _rel_l2(patched: Tensor, clean_target: Tensor) -> Tensor:
    """Per-prompt |patched − clean| / |clean|, eps-clipped denominator."""
    diff = (patched - clean_target).detach().abs()
    denom = clean_target.detach().abs().clamp_min(1e-06)
    return diff / denom

def _flip_rate(patched: Tensor, clean_target: Tensor) -> float:
    """Fraction of prompts where patched logit-diff has the same sign as clean
    AND is at least 50% of clean magnitude — i.e. the IO survives as argmax.
    """
    p = patched.detach().cpu()
    c = clean_target.detach().cpu()
    same_sign = torch.sign(p) == torch.sign(c)
    enough_mag = p.abs() >= 0.5 * c.abs()
    return float((same_sign & enough_mag).float().mean().item())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gpt2')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seeds', default='0,1,2', help='Comma-separated seeds (varies prompt sample)')
    p.add_argument('--ks', default='5,8')
    p.add_argument('--ridge', type=float, default=1.0)
    p.add_argument('--epsilon', type=float, default=0.1)
    p.add_argument('--algos', default='driver_omp,coverage_driver,top_attribution,random_matched')
    p.add_argument('--variants', default='id_test,ood_names,ood_templates,ood_long,ood_combined')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    seeds = [int(s) for s in args.seeds.split(',')]
    ks = [int(s) for s in args.ks.split(',')]
    algos = args.algos.split(',')
    variants = args.variants.split(',')
    from transformer_lens import HookedTransformer
    print(f'Loading {args.model}...')
    model = load_tl(args.model, device=args.device, revision=os.environ['SIC_MODEL_REVISION'])
    tokenizer = model.tokenizer
    candidate_nodes: List[DriverNode] = []
    for L in range(model.cfg.n_layers):
        for pos in ('END', 'IO', 'S1', 'S2'):
            for h in range(model.cfg.n_heads):
                candidate_nodes.append(DriverNode(basis='head_pos', layer=L, position=pos, component=h))
    adapter = TransformerLensAdapter(model, device=args.device)
    target = IOILogitDiff()
    print(f'  candidate_nodes={len(candidate_nodes)}  layers={model.cfg.n_layers}  heads={model.cfg.n_heads}')
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        id_clean, id_corrupt = _build_variant(tokenizer, _VARIANT_CONFIGS['id'], args.batch_size, args.device, seed=seed)
        id_test_seed = seed + 10000
        ood_batches: Dict[str, Tuple[Dict, Dict]] = {}
        ood_batches['id_test'] = _build_variant(tokenizer, _VARIANT_CONFIGS['id'], args.batch_size, args.device, seed=id_test_seed)
        for v in ('ood_names', 'ood_templates', 'ood_long', 'ood_combined'):
            if v in variants:
                ood_batches[v] = _build_variant(tokenizer, _VARIANT_CONFIGS[v], args.batch_size, args.device, seed=seed + 7777)
        disc = DriverCircuitDiscovery(adapter=adapter, target_metric=target, candidate_nodes=candidate_nodes, epsilon=args.epsilon, max_drivers=max(ks), ridge=args.ridge)
        G_id, q_id, deltas_id = disc.estimate_local_effects(id_clean, id_corrupt)
        id_clean_target = adapter.run_clean_or_corrupt(id_clean, 'clean', target).detach()
        id_corrupt_target = adapter.run_clean_or_corrupt(id_corrupt, 'corrupt', target).detach()
        print(f'\n=== seed {seed}  id_clean_margin={id_clean_target.mean():+.3f}  id_corrupt_margin={id_corrupt_target.mean():+.3f} ===')
        ood_surrogates: Dict[str, Tuple[Tensor, Tensor, Dict, Tensor, Tensor]] = {}
        for vname, (vc, vk) in ood_batches.items():
            G_v, q_v, deltas_v = disc.estimate_local_effects(vc, vk)
            cv = adapter.run_clean_or_corrupt(vc, 'clean', target).detach()
            kv = adapter.run_clean_or_corrupt(vk, 'corrupt', target).detach()
            ood_surrogates[vname] = (G_v, q_v, deltas_v, cv, kv)
            print(f'  {vname:>14s}  clean={cv.mean():+.3f}  corrupt={kv.mean():+.3f}  |q|={q_v.norm().item():.3f}')
        for kk in ks:
            for algo in algos:
                n_cand = G_id.shape[1]
                if algo == 'driver_omp':
                    sel, _, _ = orthogonal_matching_pursuit(G_id, q_id, max_drivers=kk, ridge=args.ridge)
                elif algo == 'coverage_driver':
                    sel, _, _ = coverage_driver(G_id, q_id, max_drivers=kk, coverage_target=0.95)
                elif algo == 'top_attribution':
                    sel = torch.topk((G_id.T @ q_id).abs(), min(kk, n_cand)).indices.tolist()
                elif algo == 'random_matched':
                    rng_t = torch.Generator(device='cpu').manual_seed(123 + seed)
                    sel = torch.randperm(n_cand, generator=rng_t).tolist()[:kk]
                else:
                    print(f'  skipping unknown algo {algo}')
                    continue
                if not sel:
                    continue
                drivers = [candidate_nodes[i] for i in sel]
                alpha_id = ridge_solve(G_id[:, sel], q_id, ridge=args.ridge)
                for vname in ['id_test'] + [v for v in variants if v != 'id_test']:
                    if vname not in ood_surrogates:
                        continue
                    G_v, q_v, deltas_v, cv, kv = ood_surrogates[vname]
                    patches_F = {n: float(alpha_id[j].item()) * deltas_v[n] for j, n in enumerate(drivers)}
                    patched_F = adapter.run_with_patches((kv_batch := ood_batches[vname][1]), patches_F, target).detach()
                    rels_F = _rel_l2(patched_F, cv).cpu().tolist()
                    flip_F = _flip_rate(patched_F, cv)
                    alpha_R = ridge_solve(G_v[:, sel], q_v, ridge=args.ridge)
                    patches_R = {n: float(alpha_R[j].item()) * deltas_v[n] for j, n in enumerate(drivers)}
                    patched_R = adapter.run_with_patches(kv_batch, patches_R, target).detach()
                    rels_R = _rel_l2(patched_R, cv).cpu().tolist()
                    flip_R = _flip_rate(patched_R, cv)
                    rows.append({'seed': seed, 'k': kk, 'algo': algo, 'variant': vname, 'n_drivers': len(drivers), 'drivers': [n.label() for n in drivers], 'alpha_id': [float(x) for x in alpha_id.tolist()], 'alpha_refit': [float(x) for x in alpha_R.tolist()], 'alpha_cosine': float(torch.nn.functional.cosine_similarity(alpha_id.flatten().unsqueeze(0), alpha_R.flatten().unsqueeze(0)).item()), 'alpha_l2_diff': float((alpha_id - alpha_R).norm().item()), 'clean_margin': float(cv.mean().item()), 'corrupt_margin': float(kv.mean().item()), 'rel_l2_frozen_mean': float(sum(rels_F) / len(rels_F)), 'rel_l2_frozen_max': float(max(rels_F)), 'rel_l2_refit_mean': float(sum(rels_R) / len(rels_R)), 'rel_l2_refit_max': float(max(rels_R)), 'frac_rel_gt_0p3_frozen': float(sum((1 for r in rels_F if r > 0.3)) / len(rels_F)), 'frac_rel_gt_0p3_refit': float(sum((1 for r in rels_R if r > 0.3)) / len(rels_R)), 'flip_rate_frozen': flip_F, 'flip_rate_refit': flip_R})
                    print(f"  seed={seed} k={kk:>2d} {algo:>16s}  {vname:>14s}  F-rel={rows[-1]['rel_l2_frozen_mean']:.3f}  R-rel={rows[-1]['rel_l2_refit_mean']:.3f}  αcos={rows[-1]['alpha_cosine']:.2f}  flipF={flip_F * 100:.0f}% flipR={flip_R * 100:.0f}%")
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'\nWrote {len(rows)} rows to {args.out}')
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        key = (r['algo'], r['variant'], r['k'])
        agg[key].append(r)
    print('\n=== Aggregated mean rel_l2 across seeds (Frozen-α / Refit-α) ===')
    print(f"{'algo':>16s} {'variant':>14s} {'k':>3s}  {'F-rel mean':>10s} {'R-rel mean':>10s}  {'F-flip':>7s} {'R-flip':>7s}")
    for (algo, variant, kk), recs in sorted(agg.items()):
        f_mean = sum((r['rel_l2_frozen_mean'] for r in recs)) / len(recs)
        r_mean = sum((r['rel_l2_refit_mean'] for r in recs)) / len(recs)
        f_flip = sum((r['flip_rate_frozen'] for r in recs)) / len(recs)
        r_flip = sum((r['flip_rate_refit'] for r in recs)) / len(recs)
        print(f'{algo:>16s} {variant:>14s} {kk:>3d}  {f_mean:>10.3f} {r_mean:>10.3f}  {f_flip * 100:>6.0f}% {r_flip * 100:>6.0f}%')
if __name__ == '__main__':
    main()
