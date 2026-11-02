"""
Task-specific executed experiment implementation.
"""
from __future__ import annotations
from sparse_control.runners.model_loading import load_tl, load_hf, load_tokenizer
import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import torch
from sparse_control.selection.core import DriverCircuitDiscovery, DriverNode, DriverResult, consensus_core, linearization_correlation
from sparse_control.selection.execution_context import acdc_lite_baseline, actual_gain_greedy_baseline, coverage_driver_baseline, coverage_driver_v2_baseline, eap_endpoint_nodes, eap_ig_endpoint_nodes, edge_pruning_full_baseline, edge_pruning_official_baseline, edge_pruning_proxy_baseline, endpoint_nodes_from_file, hap_official_baseline, hap_proxy_baseline, random_matched_baseline, top_node_attribution_baseline, wang2023_manual_endpoint

def resolve_device(spec: str) -> str:
    if spec == 'auto':
        if torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    return spec

def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

def build_ioi_gpt2(args: argparse.Namespace, device: str):
    from transformer_lens import HookedTransformer
    from sparse_control.adapters.transformerlens import TransformerLensAdapter, IOILogitDiff
    from sparse_control.tasks.pairs import build_ioi_pairs
    model = load_tl(args.model, device=device, revision=os.environ['SIC_MODEL_REVISION'])
    tokenizer = model.tokenizer
    clean, corrupt = build_ioi_pairs(tokenizer, n=args.batch_size, device=device, seed=args.seed)
    adapter = TransformerLensAdapter(model, device=device)
    target = IOILogitDiff()
    nodes: List[DriverNode] = []
    positions = ['END', 'IO', 'S1', 'S2']
    for layer in range(model.cfg.n_layers):
        for pos in positions:
            for head in range(model.cfg.n_heads):
                nodes.append(DriverNode(basis='head_pos', layer=layer, position=pos, component=head))
    return (adapter, target, nodes, clean, corrupt)

def build_mcqa_gemma(args: argparse.Namespace, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparse_control.adapters.huggingface import HFHookAdapter
    from sparse_control.adapters.transformerlens import MCQALogitVector
    from sparse_control.tasks.pairs import build_mcqa_pairs, MCQAExample, load_mmlu_examples
    dtype = torch.bfloat16 if device != 'cpu' else torch.float32
    tokenizer = load_tokenizer(args.model, revision=os.environ['SIC_MODEL_REVISION'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_hf(args.model, device=device, tokenizer=tokenizer, torch_dtype=dtype, attn_implementation='eager', revision=os.environ['SIC_MODEL_REVISION'])
    model.eval()
    demo_examples = [MCQAExample(question='The capital of France is', choices=['Berlin', 'Paris', 'Rome', 'Madrid'], correct_index=1, distractor_index=0), MCQAExample(question='H2O is the chemical formula for', choices=['Oxygen', 'Hydrogen', 'Water', 'Salt'], correct_index=2, distractor_index=0), MCQAExample(question='The square root of 144 is', choices=['10', '11', '12', '14'], correct_index=2, distractor_index=3), MCQAExample(question='The largest ocean on Earth is', choices=['Atlantic', 'Indian', 'Arctic', 'Pacific'], correct_index=3, distractor_index=0)]
    if getattr(args, 'mcqa_source', 'demo') == 'demo':
        examples = (demo_examples * (args.batch_size // len(demo_examples) + 1))[:args.batch_size]
    else:
        category = args.mcqa_source.replace('mmlu_', '')
        examples = load_mmlu_examples(category=category, n=args.batch_size, seed=args.seed)
    clean, corrupt = build_mcqa_pairs(tokenizer, examples, device=device)
    adapter = HFHookAdapter(model, tokenizer, device=device)
    target = MCQALogitVector()
    n_layers = model.config.num_hidden_layers
    nodes: List[DriverNode] = []
    for layer in range(n_layers):
        nodes.append(DriverNode(basis='resid_post', layer=layer, position='ANSWER'))
        nodes.append(DriverNode(basis='mlp_out', layer=layer, position='ANSWER'))
        nodes.append(DriverNode(basis='attn_out', layer=layer, position='ANSWER'))
    return (adapter, target, nodes, clean, corrupt)
PRESETS = {'ioi_gpt2': build_ioi_gpt2, 'mcqa_gemma': build_mcqa_gemma}

def summarize_result(name: str, seed: int, k: int, res: DriverResult, wallclock_s: float) -> Dict[str, Any]:
    return {'method': name, 'seed': seed, 'k_target': k, 'driver_size': len(res.drivers), 'drivers': [n.label() for n in res.drivers], 'strengths': res.strengths.detach().cpu().tolist() if res.strengths.numel() else [], 'rel_l2_to_clean': res.nonlinear_scores.get('rel_l2_to_clean'), 'mae_to_clean': res.nonlinear_scores.get('mae_to_clean'), 'linear_residual_norms': res.linear_residual_norms, 'patching_queries': sum((1 for e in res.validation_log if e.get('action', '').startswith(('try_remove', 'fallback_', 'actual_gain', 'initial_validate', 'final_validate')))), 'wallclock_s': wallclock_s}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--preset', required=True, choices=list(PRESETS))
    p.add_argument('--model', default=None, help='Model id (preset default if omitted)')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seeds', type=str, default='0,1,2')
    p.add_argument('--seed', type=int, default=0, help='Seed passed into data loaders')
    p.add_argument('--ks', type=str, default='3,5,8,12')
    p.add_argument('--epsilon', type=float, default=0.1)
    p.add_argument('--ridge', type=float, default=1e-05)
    p.add_argument('--lin-tol', type=float, default=0.001)
    p.add_argument('--max-prune-queries', type=int, default=64)
    p.add_argument('--target-sketch-dim', type=int, default=None, help='Gaussian sketch dim for vector targets; None disables.')
    p.add_argument('--fallback', default='actual_gain', choices=['actual_gain', 'omp_residual', 'none'])
    p.add_argument('--ridges', type=str, default=None, help="Comma-separated ridge sweep, e.g. '1e-5,1e-3,1e-2,1e-1,1'. If set, only Driver-OMP is re-run for each ridge (baselines use the single --ridge value).")
    p.add_argument('--mcqa-source', default='demo', choices=['demo', 'mmlu_stem', 'mmlu_humanities', 'mmlu_other', 'mmlu_all'], help='MCQA prompt source for the mcqa_gemma preset.')
    p.add_argument('--include-sota', action='store_true', help='Include EAP-IG / ACDC-lite / Wang2023 manual / EP-proxy / HAP-proxy SOTA baselines.')
    p.add_argument('--ep-lambda-l0', type=float, default=0.05, help='L0 sparsity weight for edge_pruning_proxy / hap_proxy / edge_pruning_full.')
    p.add_argument('--hap-pre-filter', type=int, default=64, help='EAP pre-filter top-K for HAP proxy stage 1.')
    p.add_argument('--include-ep-full', action='store_true', help='Also run edge_pruning_full (mask trained through real transformer forward).')
    p.add_argument('--ep-full-steps', type=int, default=200, help='Training steps for edge_pruning_full mask optimisation.')
    p.add_argument('--ep-flavor', default='proxy', choices=['proxy', 'official'], help="Edge-Pruning gate flavor: 'proxy' = sigmoid mask, 'official' = hard-concrete (Louizos 2018).")
    p.add_argument('--hap-flavor', default='proxy', choices=['proxy', 'official'], help="HAP flavor: 'proxy' = sigmoid mask, 'official' = hard-concrete.")
    p.add_argument('--ep-official-steps', type=int, default=1000, help='Training steps for hard-concrete edge_pruning_official.')
    p.add_argument('--ep-official-mc', type=int, default=4, help='MC samples per step for hard-concrete training.')
    p.add_argument('--coverage-target', type=float, default=0.95, help='Coverage-Driver early-stop weighted-coverage threshold.')
    p.add_argument('--alpha-clip', type=float, default=None, help='If set, clip each solved strength to [-alpha_clip, alpha_clip] before nonlinear validation. Mitigates linear overshoot on highly-collinear candidate bases.')
    p.add_argument('--eap-edges', type=int, default=32, help='Edge budget for the EAP endpoint baseline.')
    p.add_argument('--edge-pruning-json', default=None)
    p.add_argument('--hap-json', default=None)
    p.add_argument('--skip-consensus', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.model is None:
        args.model = {'ioi_gpt2': 'gpt2', 'mcqa_gemma': 'google/gemma-2-2b', 'refusal_gemma': 'google/gemma-2-2b', 'greater_than_gpt2': 'gpt2'}[args.preset]
    device = resolve_device(args.device)
    seeds = [int(s) for s in args.seeds.split(',')]
    ks = [int(k) for k in args.ks.split(',')]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = out_path.open('w', encoding='utf-8')
    all_driver_omp_results: List[DriverResult] = []
    try:
        for seed in seeds:
            set_determinism(seed)
            args.seed = seed
            adapter, target, nodes, clean, corrupt = PRESETS[args.preset](args, device)
            for k in ks:
                disc = DriverCircuitDiscovery(adapter=adapter, target_metric=target, candidate_nodes=nodes, epsilon=args.epsilon, max_drivers=k, ridge=args.ridge, lin_tol=args.lin_tol, target_sketch_dim=args.target_sketch_dim, sketch_seed=seed, fallback_mode=args.fallback, max_prune_queries=args.max_prune_queries, alpha_clip=args.alpha_clip)
                t0 = time.time()
                res = disc.run(clean, corrupt)
                rec = summarize_result('driver_omp', seed, k, res, time.time() - t0)
                rec['ridge'] = args.ridge
                log_f.write(json.dumps(rec) + '\n')
                all_driver_omp_results.append(res)
                if args.ridges:
                    for ridge in [float(r) for r in args.ridges.split(',')]:
                        if ridge == args.ridge:
                            continue
                        disc_r = DriverCircuitDiscovery(adapter=adapter, target_metric=target, candidate_nodes=nodes, epsilon=args.epsilon, max_drivers=k, ridge=ridge, lin_tol=args.lin_tol, target_sketch_dim=args.target_sketch_dim, sketch_seed=seed, fallback_mode=args.fallback, max_prune_queries=args.max_prune_queries)
                        t0 = time.time()
                        res_r = disc_r.run(clean, corrupt)
                        rec_r = summarize_result(f'driver_omp_ridge', seed, k, res_r, time.time() - t0)
                        rec_r['ridge'] = ridge
                        log_f.write(json.dumps(rec_r) + '\n')
                t0 = time.time()
                res_cov = coverage_driver_baseline(disc, clean, corrupt, k=k, coverage_target=args.coverage_target)
                rec_cov = summarize_result('coverage_driver', seed, k, res_cov, time.time() - t0)
                rec_cov['ridge'] = args.ridge
                hist = res_cov.validation_log[0].get('coverage_history') if res_cov.validation_log else None
                if hist is not None:
                    rec_cov['coverage_history'] = hist
                log_f.write(json.dumps(rec_cov) + '\n')
                t0 = time.time()
                res_cv2 = coverage_driver_v2_baseline(disc, clean, corrupt, k=k, coverage_target=0.98, weight_mode='uniform')
                rec_cv2 = summarize_result('coverage_driver_v2', seed, k, res_cv2, time.time() - t0)
                rec_cv2['ridge'] = args.ridge
                log_f.write(json.dumps(rec_cv2) + '\n')
                t0 = time.time()
                res_top = top_node_attribution_baseline(disc, clean, corrupt, k=k)
                log_f.write(json.dumps(summarize_result('top_attribution', seed, k, res_top, time.time() - t0)) + '\n')
                t0 = time.time()
                res_rand = random_matched_baseline(disc, clean, corrupt, k=k, seed=seed)
                log_f.write(json.dumps(summarize_result('random_matched', seed, k, res_rand, time.time() - t0)) + '\n')
                t0 = time.time()
                res_ag = actual_gain_greedy_baseline(disc, clean, corrupt, k=k, topk_proposal=32)
                log_f.write(json.dumps(summarize_result('actual_gain_greedy', seed, k, res_ag, time.time() - t0)) + '\n')
                t0 = time.time()
                res_eap = eap_endpoint_nodes(disc, clean, corrupt, num_edges=args.eap_edges)
                log_f.write(json.dumps(summarize_result('eap_endpoint', seed, k, res_eap, time.time() - t0)) + '\n')
                if args.include_sota:
                    t0 = time.time()
                    res_eapig = eap_ig_endpoint_nodes(disc, clean, corrupt, num_edges=args.eap_edges)
                    log_f.write(json.dumps(summarize_result('eap_ig_endpoint', seed, k, res_eapig, time.time() - t0)) + '\n')
                    t0 = time.time()
                    res_acdc = acdc_lite_baseline(disc, clean, corrupt, k=k, epsilon=0.1, topk_pool=32)
                    log_f.write(json.dumps(summarize_result('acdc_lite', seed, k, res_acdc, time.time() - t0)) + '\n')
                    if args.preset == 'ioi_gpt2':
                        t0 = time.time()
                        res_wang = wang2023_manual_endpoint(disc, clean, corrupt, position='END')
                        log_f.write(json.dumps(summarize_result('wang2023_manual', seed, k, res_wang, time.time() - t0)) + '\n')
                    t0 = time.time()
                    if args.ep_flavor == 'official':
                        res_ep = edge_pruning_official_baseline(disc, clean, corrupt, k=k, n_steps=args.ep_official_steps, lambda_l0=args.ep_lambda_l0, n_mc_samples=args.ep_official_mc)
                        ep_label = 'edge_pruning_official'
                    else:
                        res_ep = edge_pruning_proxy_baseline(disc, clean, corrupt, k=k, lambda_l0=args.ep_lambda_l0)
                        ep_label = 'edge_pruning_proxy'
                    log_f.write(json.dumps(summarize_result(ep_label, seed, k, res_ep, time.time() - t0)) + '\n')
                    t0 = time.time()
                    if args.hap_flavor == 'official':
                        res_hap = hap_official_baseline(disc, clean, corrupt, k=k, pre_filter_top=args.hap_pre_filter, n_steps=max(400, args.ep_official_steps // 2), lambda_l0=args.ep_lambda_l0, n_mc_samples=args.ep_official_mc)
                        hap_label = 'hap_official'
                    else:
                        res_hap = hap_proxy_baseline(disc, clean, corrupt, k=k, pre_filter_top=args.hap_pre_filter, lambda_l0=args.ep_lambda_l0)
                        hap_label = 'hap_proxy'
                    log_f.write(json.dumps(summarize_result(hap_label, seed, k, res_hap, time.time() - t0)) + '\n')
                if args.preset in ('ioi_gpt2', 'greater_than_gpt2') and args.include_ep_full:
                    t0 = time.time()
                    res_ep_full = edge_pruning_full_baseline(disc, clean, corrupt, k=k, n_steps=args.ep_full_steps, lambda_l0=args.ep_lambda_l0)
                    log_f.write(json.dumps(summarize_result('edge_pruning_full', seed, k, res_ep_full, time.time() - t0)) + '\n')
                for label, path in (('edge_pruning', args.edge_pruning_json), ('hap', args.hap_json)):
                    if path is None:
                        continue
                    t0 = time.time()
                    res_ext = endpoint_nodes_from_file(disc, path, clean, corrupt, method_name=label)
                    log_f.write(json.dumps(summarize_result(label, seed, k, res_ext, time.time() - t0)) + '\n')
                log_f.flush()
        if not args.skip_consensus and all_driver_omp_results:
            core, backup, freq = consensus_core(all_driver_omp_results, gamma=0.7)
            consensus_record = {'record_type': 'consensus', 'gamma': 0.7, 'core': [n.label() for n in core], 'backup': [n.label() for n in backup], 'freq': {n.label(): p for n, p in freq.items()}, 'num_runs': len(all_driver_omp_results)}
            log_f.write(json.dumps(consensus_record) + '\n')
    finally:
        log_f.close()
    print(f'Wrote {out_path}')
if __name__ == '__main__':
    main()
