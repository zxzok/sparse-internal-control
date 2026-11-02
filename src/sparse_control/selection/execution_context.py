"""
Numerical comparison kernels used by the complete nominal execution sequence. Public paper comparisons are exported from baselines.py.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import torch
from torch import Tensor
from sparse_control.selection.core import DriverCircuitDiscovery, DriverNode, DriverResult, HookAdapter, TargetMetric, actual_gain_greedy, coverage_driver, coverage_score_matrix, ridge_solve

def random_matched_nodes(candidate_nodes: Sequence[DriverNode], k: int, seed: int=0) -> List[DriverNode]:
    rng = random.Random(seed)
    k = min(k, len(candidate_nodes))
    return rng.sample(list(candidate_nodes), k)

def top_node_attribution_baseline(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], k: int) -> DriverResult:
    """Top-k drivers by absolute linear attribution ``|G_v^T q|`` with
    ridge-solved joint strengths on the validation pass."""
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    scores = torch.abs(G.T @ q)
    k = min(k, G.shape[1])
    selected = torch.topk(scores, k).indices.tolist()
    drivers = [discovery.candidate_nodes[i] for i in selected]
    strengths = ridge_solve(G[:, selected], q, ridge=discovery.ridge)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    _, final_scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, deltas, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=final_scores, validation_log=[{'action': 'top_attribution_validate', **final_scores}], selected_indices=selected, method='top_node_attribution')

def actual_gain_greedy_baseline(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], k: int, topk_proposal: int=32) -> DriverResult:
    """Thin wrapper around ``core.actual_gain_greedy`` for consistency with
    the baseline-registry style used in the experiment runner."""
    return actual_gain_greedy(discovery, clean_batch=clean_batch, corrupt_batch=corrupt_batch, max_drivers=k, topk_proposal=topk_proposal)

def random_matched_baseline(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], k: int, seed: int=0) -> DriverResult:
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    drivers = random_matched_nodes(discovery.candidate_nodes, k, seed=seed)
    selected = [discovery.candidate_nodes.index(d) for d in drivers]
    strengths = ridge_solve(G[:, selected], q, ridge=discovery.ridge) if selected else torch.empty(0, device=q.device, dtype=q.dtype)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    _, scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, deltas, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores, validation_log=[{'action': 'random_validate', **scores}], selected_indices=selected, method='random_matched')

def coverage_driver_baseline(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], k: int, coverage_target: float=0.95, weight_mode: str='absq') -> DriverResult:
    """Coverage-Driver: greedy submodular minimum-cover selection + ridge refit.

    Pipeline
    --------
    1. Estimate G, q, deltas via `discovery.estimate_local_effects` (same call
       Driver-OMP and the linear baselines use, so wallclock is comparable).
    2. Run `core.coverage_driver` on G, q to greedily select up to `k` driver
       nodes that maximise weighted soft coverage Σ_i w_i · min(1, Σ a_iv).
    3. Refit driver strengths on the selected subset with ridge regression
       — this lets selected drivers carry continuous (and possibly negative)
       alphas during nonlinear validation, matching the Driver-OMP / top-attr
       protocol.
    4. Validate with the same `discovery.validate_driver_set` patching call.

    The validation log records the per-pick weighted-coverage trajectory so
    coverage curves can be plotted post-hoc against driver-set size.
    """
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    selected, coverage_history, marginal_gains = coverage_driver(G, q, max_drivers=k, coverage_target=coverage_target, weight_mode=weight_mode)
    drivers = [discovery.candidate_nodes[i] for i in selected]
    strengths = ridge_solve(G[:, selected], q, ridge=discovery.ridge) if selected else torch.empty(0, device=q.device, dtype=q.dtype)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    _, scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: deltas[n] for n in drivers}, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores, validation_log=[{'action': 'coverage_driver_validate', 'coverage_history': coverage_history, 'marginal_gains': marginal_gains, **scores}], selected_indices=selected, method='coverage_driver')

def eap_endpoint_nodes(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], num_edges: int) -> DriverResult:
    """EAP-flavoured baseline that picks ``num_edges`` edges by
    ``|G_v^T q| * |G_w^T q|`` surrogates, then converts endpoints to a node
    set. This matches the "convert edges to node sets" protocol in Phase 5
    of the experiment plan.

    Because Driver-OMP already estimates single-node linear effects, EAP
    endpoints are approximated here as the unique nodes covered by the
    top ``num_edges`` (u, v) pairs ranked by ``|s_u| * |s_v|`` with
    ``s_v = G_v^T q``.
    """
    G, q, _deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    scores = torch.abs(G.T @ q)
    n = int(scores.shape[0])
    pair_scores = scores.unsqueeze(0) * scores.unsqueeze(1)
    pair_scores.fill_diagonal_(0.0)
    flat = pair_scores.view(-1)
    top_flat = torch.topk(flat, min(num_edges, flat.numel())).indices.tolist()
    endpoints: List[int] = []
    seen = set()
    for idx in top_flat:
        u, v = (idx // n, idx % n)
        for e in (u, v):
            if e not in seen:
                seen.add(e)
                endpoints.append(int(e))
    drivers = [discovery.candidate_nodes[i] for i in endpoints]
    strengths = ridge_solve(G[:, endpoints], q, ridge=discovery.ridge) if endpoints else torch.empty(0, device=q.device, dtype=q.dtype)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    _, scores_v = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: _deltas[n] for n in drivers}, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores_v, validation_log=[{'action': 'eap_endpoint_validate', **scores_v}], selected_indices=endpoints, method='eap_endpoint')

def eap_ig_endpoint_nodes(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], num_edges: int, n_ig_steps: int=2) -> DriverResult:
    """EAP-IG: Integrated-Gradients-flavoured EAP endpoint baseline.

    Trapezoid-rule IG with `n_ig_steps=2`: average jacobians at corrupt and at
    clean baselines. Cheaper than full IG (N≥10) but already strictly more
    faithful than vanilla EAP which uses only the corrupt-side jacobian.
    """
    G_corr, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    G_clean, q_neg, _ = discovery.estimate_local_effects(corrupt_batch, clean_batch)
    s_corr = G_corr.T @ q
    s_clean = G_clean.T @ q_neg
    s_ig = 0.5 * (s_corr - s_clean)
    abs_scores = s_ig.abs()
    n = int(abs_scores.shape[0])
    pair_scores = abs_scores.unsqueeze(0) * abs_scores.unsqueeze(1)
    pair_scores.fill_diagonal_(0.0)
    flat = pair_scores.view(-1)
    top_flat = torch.topk(flat, min(num_edges, flat.numel())).indices.tolist()
    endpoints: List[int] = []
    seen = set()
    for idx in top_flat:
        u, v = (idx // n, idx % n)
        for e in (u, v):
            if e not in seen:
                seen.add(e)
                endpoints.append(int(e))
    drivers = [discovery.candidate_nodes[i] for i in endpoints]
    strengths = ridge_solve(G_corr[:, endpoints], q, ridge=discovery.ridge) if endpoints else torch.empty(0, device=q.device, dtype=q.dtype)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    _, scores_v = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: deltas[n] for n in drivers}, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores_v, validation_log=[{'action': 'eap_ig_endpoint_validate', 'n_ig_steps': n_ig_steps, **scores_v}], selected_indices=endpoints, method='eap_ig_endpoint')
_WANG_2023_IOI_HEADS = [(2, 2), (4, 11), (0, 1), (0, 10), (3, 0), (5, 5), (5, 8), (5, 9), (6, 9), (7, 3), (7, 9), (8, 6), (8, 10), (9, 6), (9, 9), (10, 0), (10, 10), (11, 2), (10, 7), (11, 10)]

def wang2023_manual_endpoint(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], position: str='END') -> DriverResult:
    """Hand-curated IOI circuit from Wang et al. 2023, Table 2.
    All heads patched at `position` (default END)."""
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    label_to_idx = {n.label(): i for i, n in enumerate(discovery.candidate_nodes)}
    endpoints: List[int] = []
    for layer, head in _WANG_2023_IOI_HEADS:
        label = f'head_pos:L{layer}:P{position}:H{head}'
        if label in label_to_idx:
            endpoints.append(label_to_idx[label])
    drivers = [discovery.candidate_nodes[i] for i in endpoints]
    strengths = ridge_solve(G[:, endpoints], q, ridge=discovery.ridge) if endpoints else torch.empty(0, device=q.device, dtype=q.dtype)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    _, scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: deltas[n] for n in drivers}, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores, validation_log=[{'action': 'wang2023_manual_validate', 'n_heads': len(endpoints), **scores}], selected_indices=endpoints, method='wang2023_manual')

def acdc_lite_baseline(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], k: int, epsilon: float=0.1, topk_pool: int=32, max_queries: int=96) -> DriverResult:
    """Node-level ACDC-lite: backward greedy elimination from a top-k pool.

    Initialises the candidate set with `topk_pool` highest-attribution nodes,
    then attempts to drop the lowest-attribution node first; if the validated
    rel_l2 stays within `epsilon` of the running best, the drop is accepted.
    Stops at |D| ≤ k, no further accepts, or `max_queries`.
    """
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    n_cand = G.shape[1]
    scores = (G.T @ q).abs()
    pool_size = min(topk_pool, n_cand)
    pool = torch.topk(scores, pool_size).indices.tolist()
    pool_scores = scores[pool].tolist()
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)

    def _try_set(idxs: List[int]):
        if not idxs:
            return (float('inf'), {})
        drivers = [discovery.candidate_nodes[i] for i in idxs]
        strengths = ridge_solve(G[:, idxs], q, ridge=discovery.ridge)
        _, sv = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: deltas[n] for n in drivers}, strengths)
        return (sv['rel_l2_to_clean'], sv)
    keep = list(pool)
    queries = 0
    base_rel, _ = _try_set(keep)
    queries += 1
    removal_order = sorted(range(len(pool)), key=lambda i: pool_scores[i])
    removal_order = [pool[i] for i in removal_order]
    log: List[Dict[str, Any]] = [{'action': 'acdc_init', 'n': len(keep), 'rel_l2': base_rel}]
    for cand in removal_order:
        if queries >= max_queries:
            break
        if len(keep) <= k:
            break
        if cand not in keep:
            continue
        trial = [i for i in keep if i != cand]
        trial_rel, _ = _try_set(trial)
        queries += 1
        if trial_rel <= base_rel + epsilon:
            keep = trial
            base_rel = min(base_rel, trial_rel)
            log.append({'action': 'acdc_drop', 'node': discovery.candidate_nodes[cand].label(), 'rel_l2': trial_rel, 'remaining': len(keep)})
    drivers = [discovery.candidate_nodes[i] for i in keep]
    strengths = ridge_solve(G[:, keep], q, ridge=discovery.ridge) if keep else torch.empty(0, device=q.device, dtype=q.dtype)
    _, final_scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: deltas[n] for n in drivers}, strengths)
    log.append({'action': 'acdc_final', 'n': len(keep), **final_scores, 'queries': queries})
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=final_scores, validation_log=log, selected_indices=keep, method='acdc_lite')

def _train_l0_node_mask(G: torch.Tensor, q: torch.Tensor, n_steps: int=400, lambda_l0: float=0.05, lr: float=0.05, temp: float=0.5, init_logit: float=0.0, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Train a soft L0-style binary mask m ∈ (0,1)^k jointly with α.

    Optimisation:
        m = sigmoid(z / temp)
        loss = ||q - G (m ⊙ α)||² + λ_L0 · Σ m
    Returns (m_final, α_final) after gradient descent.
    """
    n, k = G.shape
    if device is None:
        device = G.device
    z = torch.full((k,), float(init_logit), device=device, dtype=G.dtype, requires_grad=True)
    a = torch.zeros(k, device=device, dtype=G.dtype, requires_grad=True)
    opt = torch.optim.Adam([z, a], lr=lr)
    for _ in range(n_steps):
        m = torch.sigmoid(z / max(temp, 0.001))
        recon = q - G @ (m * a)
        loss = (recon * recon).sum() + lambda_l0 * m.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        m_final = torch.sigmoid(z / max(temp, 0.001))
    return (m_final.detach(), a.detach())

def coverage_driver_v2_baseline(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], k: int, coverage_target: float=0.98, weight_mode: str='uniform') -> DriverResult:
    """Coverage-Driver v2: ablation-found best config (weight=uniform, target=0.98).

    Same submodular pipeline as `coverage_driver_baseline` but with the two
    hyperparameters that the §8.3 ablation identified as Pareto-better.
    """
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    selected, coverage_history, marginal_gains = coverage_driver(G, q, max_drivers=k, coverage_target=coverage_target, weight_mode=weight_mode)
    drivers = [discovery.candidate_nodes[i] for i in selected]
    strengths = ridge_solve(G[:, selected], q, ridge=discovery.ridge) if selected else torch.empty(0, device=q.device, dtype=q.dtype)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    _, scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: deltas[n] for n in drivers}, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores, validation_log=[{'action': 'coverage_driver_v2_validate', 'weight_mode': weight_mode, 'coverage_target': coverage_target, 'coverage_history': coverage_history, 'marginal_gains': marginal_gains, **scores}], selected_indices=selected, method='coverage_driver_v2')

def edge_pruning_full_baseline(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], k: int, n_steps: int=200, lambda_l0: float=0.02, lr: float=0.1, temp: float=0.5) -> DriverResult:
    """Edge-Pruning-style L0 mask training in the FULL non-linear transformer.

    Unlike `edge_pruning_proxy_baseline` which trains a mask on the linear
    surrogate G·(m⊙α) ≈ q, this baseline trains a per-candidate soft mask
    `m_v = σ(z_v / τ)` and runs the actual transformer's `run_with_patches`
    every step, taking gradients through the model's full non-linear graph.

    Per step:
        m = sigmoid(z / temp)
        patches = {v: m_v · delta_v for v in candidates}
        target = M_target(model with patches on corrupt)
        loss = ||target - target_clean||² + λ_L0 · Σ m
        backprop to update z

    At end, threshold m to top-k.

    This is the closest in-framework approximation of official Edge Pruning
    (Bhaskar et al. 2024 NeurIPS) within our adapter. Full official Edge
    Pruning trains masks per-edge in the transformer's edge graph; we train
    masks per-node in the same node basis used by Driver-OMP. The KEY
    difference from `edge_pruning_proxy_baseline` is that we use the real
    non-linear forward pass (vs the linear surrogate), giving the optimisation
    access to all the information official Edge Pruning would.
    """
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    n_cand = len(discovery.candidate_nodes)
    device = q.device
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric).detach()
    z = torch.zeros(n_cand, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for step in range(n_steps):
        m = torch.sigmoid(z / max(temp, 0.001))
        patches: Dict[DriverNode, torch.Tensor] = {}
        for i, node in enumerate(discovery.candidate_nodes):
            if float(m[i].detach().item()) < 0.0001:
                continue
            d = deltas[node]
            patches[node] = m[i].to(d.dtype) * d
        if not patches:
            break
        target_pred = _run_with_patches_grad(discovery.adapter, corrupt_batch, patches, discovery.target_metric)
        if target_pred.ndim == 1:
            recon = target_pred - clean_target
        else:
            recon = (target_pred - clean_target).reshape(target_pred.shape[0], -1)
        loss = (recon * recon).mean() + lambda_l0 * m.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        m_final = torch.sigmoid(z / max(temp, 0.001)).detach()
    top_k = int(min(k, n_cand))
    sel = torch.topk(m_final, top_k).indices.tolist()
    drivers = [discovery.candidate_nodes[i] for i in sel]
    strengths = ridge_solve(G[:, sel], q, ridge=discovery.ridge)
    _, scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: deltas[n] for n in drivers}, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores, validation_log=[{'action': 'edge_pruning_full_validate', 'n_steps': n_steps, 'lambda_l0': lambda_l0, 'mask_l1': float(m_final.sum().item()), **scores}], selected_indices=sel, method='edge_pruning_full')

def _run_with_patches_grad(adapter, batch, patches, target_metric):
    """Like adapter.run_with_patches but WITHOUT no_grad — gradient flows
    through patch tensors so we can backprop through soft masks.
    Specific to TransformerLensAdapter (uses model.run_with_hooks).
    """
    from sparse_control.adapters.transformerlens import TransformerLensAdapter
    if not isinstance(adapter, TransformerLensAdapter):
        return adapter.run_with_patches(batch, patches, target_metric)
    tokens = batch['tokens'].to(adapter.device)
    patches_by_hook: Dict[str, List] = {}
    for node, delta in patches.items():
        patches_by_hook.setdefault(adapter.hook_name(node), []).append((node, delta))

    def make_hook(name):

        def patch_hook(act, hook):
            out = act
            for node, delta in patches_by_hook[name]:
                out = adapter._scatter_into_full(out, node, delta, alpha=1.0, batch=batch)
            return out
        return patch_hook
    hooks = [(name, make_hook(name)) for name in patches_by_hook]
    logits = adapter.model.run_with_hooks(tokens, fwd_hooks=hooks)
    return target_metric(logits, batch)

def edge_pruning_proxy_baseline(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], k: int, lambda_l0: float=0.05, n_steps: int=400) -> DriverResult:
    """Edge-Pruning-style proxy: learn an L0-regularised soft binary mask
    over candidate nodes via gradient descent on the linear reconstruction
    objective, then keep the top-k by mask probability.

    This captures the SPIRIT of Edge Pruning (continuous relaxation of an
    L0 penalty trained to optimise faithfulness) but at node level on our
    linear surrogate. Strict Edge Pruning would train edge masks within the
    transformer's actual computation graph; we approximate that with the
    G, q linear system.
    """
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    m, _ = _train_l0_node_mask(G, q, n_steps=n_steps, lambda_l0=lambda_l0)
    top_k = int(min(k, G.shape[1]))
    sel = torch.topk(m, top_k).indices.tolist()
    drivers = [discovery.candidate_nodes[i] for i in sel]
    strengths = ridge_solve(G[:, sel], q, ridge=discovery.ridge)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    _, scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: deltas[n] for n in drivers}, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores, validation_log=[{'action': 'edge_pruning_proxy_validate', 'lambda_l0': lambda_l0, 'n_steps': n_steps, 'mask_l1': float(m.sum().item()), **scores}], selected_indices=sel, method='edge_pruning_proxy')

def hap_proxy_baseline(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], k: int, pre_filter_top: int=64, lambda_l0: float=0.05, n_steps: int=300) -> DriverResult:
    """HAP-style proxy: 2-stage pipeline.
      Stage 1 (Hybrid Attribution): filter to top-`pre_filter_top` candidates
              by EAP attribution score |G^T q|.
      Stage 2 (Pruning):           train an L0-regularised mask within the
              filtered subset, keep the top-k by mask probability.

    This mirrors HAP's structural pattern (attribution-then-pruning) within
    our linear surrogate. Strict HAP requires running the official EAP +
    edge mask training pipeline.
    """
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    eap_scores = (G.T @ q).abs()
    pool = torch.topk(eap_scores, min(pre_filter_top, G.shape[1])).indices.tolist()
    G_sub = G[:, pool]
    m_sub, _ = _train_l0_node_mask(G_sub, q, n_steps=n_steps, lambda_l0=lambda_l0)
    top_k = int(min(k, len(pool)))
    sub_sel = torch.topk(m_sub, top_k).indices.tolist()
    sel = [pool[i] for i in sub_sel]
    drivers = [discovery.candidate_nodes[i] for i in sel]
    strengths = ridge_solve(G[:, sel], q, ridge=discovery.ridge)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    _, scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: deltas[n] for n in drivers}, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores, validation_log=[{'action': 'hap_proxy_validate', 'pre_filter_top': pre_filter_top, 'lambda_l0': lambda_l0, 'n_steps': n_steps, 'mask_l1': float(m_sub.sum().item()), **scores}], selected_indices=sel, method='hap_proxy')

def _hard_concrete_sample(z: torch.Tensor, beta: float, gamma: float, zeta: float, deterministic: bool=False) -> torch.Tensor:
    """Hard-concrete reparameterisation (Louizos et al. 2018, Eq. 10–11).

    During training:    u ~ U(eps, 1−eps);  s = σ((log u − log(1−u) + z)/β)
    During eval:        s = σ(z) (deterministic).
    Stretch & clamp:    s_bar = s · (ζ − γ) + γ;  m = clamp(s_bar, 0, 1)
    """
    if deterministic:
        s = torch.sigmoid(z)
    else:
        eps = 1e-06
        u = torch.empty_like(z).uniform_(eps, 1.0 - eps)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + z) / max(beta, 0.001))
    s_bar = s * (zeta - gamma) + gamma
    return s_bar.clamp(0.0, 1.0)

def _expected_l0(z: torch.Tensor, beta: float, gamma: float, zeta: float) -> torch.Tensor:
    """E[||m||_0] under the hard-concrete distribution — closed form.

        P(m_i > 0) = σ(z_i − β · log(−γ/ζ))
    """
    import math
    shift = beta * math.log(-gamma / zeta)
    return torch.sigmoid(z - shift).sum()

def _train_hard_concrete_node_mask(G: torch.Tensor, q: torch.Tensor, n_steps: int=1000, lambda_l0_final: float=0.05, lr: float=0.05, beta_start: float=2.0 / 3.0, beta_end: float=1.0 / 12.0, gamma: float=-0.1, zeta: float=1.1, init_logit: float=1.0, warmup_frac: float=0.3, n_mc_samples: int=4, device=None) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
    """Train a hard-concrete L0 mask on the linear surrogate G·(m⊙α) ≈ q.

    Differs from `_train_l0_node_mask` in three ways:
      1. Stochastic hard-concrete reparameterisation (vs deterministic σ(z/τ))
      2. Closed-form expected L0 (vs Σ σ(z/τ))
      3. Temperature β + λ_L0 annealing schedules

    Returns (m_eval, α, history) where ``m_eval`` is the deterministic
    eval-time gate values and ``history`` is per-100-step diagnostics.
    """
    n, k = G.shape
    if device is None:
        device = G.device
    dtype = G.dtype
    z = torch.full((k,), float(init_logit), device=device, dtype=dtype, requires_grad=True)
    a = torch.zeros(k, device=device, dtype=dtype, requires_grad=True)
    opt = torch.optim.Adam([z, a], lr=lr)
    history: List[Dict[str, float]] = []
    warmup_steps = int(warmup_frac * n_steps)
    for step in range(n_steps):
        progress = step / max(1, n_steps - 1)
        beta_t = beta_start * (beta_end / beta_start) ** progress
        if step < warmup_steps:
            lam_t = lambda_l0_final * (step / max(1, warmup_steps))
        else:
            lam_t = lambda_l0_final
        recon_loss = 0.0
        for _ in range(n_mc_samples):
            m = _hard_concrete_sample(z, beta_t, gamma, zeta, deterministic=False)
            recon = q - G @ (m * a)
            recon_loss = recon_loss + (recon * recon).sum()
        recon_loss = recon_loss / n_mc_samples
        l0 = _expected_l0(z, beta_t, gamma, zeta)
        loss = recon_loss + lam_t * l0
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            with torch.no_grad():
                m_eval = _hard_concrete_sample(z, beta_t, gamma, zeta, deterministic=True)
                history.append({'step': step, 'beta': beta_t, 'lambda_l0': lam_t, 'recon_loss': float(recon_loss.detach().item()), 'expected_l0': float(l0.detach().item()), 'n_active_eval': int((m_eval > 0.5).sum().item()), 'mean_m_eval': float(m_eval.mean().item())})
    with torch.no_grad():
        m_eval = _hard_concrete_sample(z, beta_end, gamma, zeta, deterministic=True).detach()
    return (m_eval, a.detach(), history)

def edge_pruning_official_baseline(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], k: int, n_steps: int=1000, lambda_l0: float=0.05, lr: float=0.05, n_mc_samples: int=4) -> DriverResult:
    """Faithful Edge-Pruning baseline using the hard-concrete distribution.

    Closer to Bhaskar et al. 2024 NeurIPS than ``edge_pruning_proxy_baseline``:
    uses the canonical Louizos 2018 stretched-sigmoid gate, closed-form
    expected L0 penalty, β-annealing, and λ_L0 warmup.

    Still operates on the *linear* surrogate G·(m⊙α) ≈ q (vs the official
    repo which trains gates inside the transformer's edge graph).  For full
    in-graph training see ``edge_pruning_official_full_baseline`` below.
    """
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    m, _, history = _train_hard_concrete_node_mask(G, q, n_steps=n_steps, lambda_l0_final=lambda_l0, lr=lr, n_mc_samples=n_mc_samples)
    top_k = int(min(k, G.shape[1]))
    eap = (G.T @ q).abs()
    score = m + 0.001 * eap / (eap.max() + 1e-09)
    sel = torch.topk(score, top_k).indices.tolist()
    drivers = [discovery.candidate_nodes[i] for i in sel]
    strengths = ridge_solve(G[:, sel], q, ridge=discovery.ridge)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    _, scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: deltas[n] for n in drivers}, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores, validation_log=[{'action': 'edge_pruning_official_validate', 'n_steps': n_steps, 'lambda_l0': lambda_l0, 'n_mc_samples': n_mc_samples, 'mask_active_at_eval': int((m > 0.5).sum().item()), 'training_history': history, **scores}], selected_indices=sel, method='edge_pruning_official')

def edge_pruning_official_full_baseline(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], k: int, n_steps: int=300, lambda_l0: float=0.02, lr: float=0.1, beta_start: float=2.0 / 3.0, beta_end: float=1.0 / 12.0, gamma: float=-0.1, zeta: float=1.1, init_logit: float=1.0, warmup_frac: float=0.3) -> DriverResult:
    """In-graph faithful EP: hard-concrete gates trained through the actual
    transformer (combines `edge_pruning_full_baseline` with hard-concrete).

    For each step:
        m ~ HardConcrete(z, β_t, γ, ζ)
        patches = {v: m_v · delta_v}
        target = M_target(model with patches on corrupt)
        loss = ||target − target_clean||² + λ_L0 · E[||m||_0]
    """
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    n_cand = len(discovery.candidate_nodes)
    device = q.device
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric).detach()
    z = torch.full((n_cand,), float(init_logit), device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    warmup_steps = int(warmup_frac * n_steps)
    history: List[Dict[str, float]] = []
    for step in range(n_steps):
        progress = step / max(1, n_steps - 1)
        beta_t = beta_start * (beta_end / beta_start) ** progress
        lam_t = lambda_l0 * min(1.0, step / max(1, warmup_steps))
        m = _hard_concrete_sample(z, beta_t, gamma, zeta, deterministic=False)
        patches: Dict[DriverNode, torch.Tensor] = {}
        for i, node in enumerate(discovery.candidate_nodes):
            if float(m[i].detach().item()) < 0.0001:
                continue
            d = deltas[node]
            patches[node] = m[i].to(d.dtype) * d
        if not patches:
            continue
        target_pred = _run_with_patches_grad(discovery.adapter, corrupt_batch, patches, discovery.target_metric)
        if target_pred.ndim == 1:
            recon = target_pred - clean_target
        else:
            recon = (target_pred - clean_target).reshape(target_pred.shape[0], -1)
        l0 = _expected_l0(z, beta_t, gamma, zeta)
        loss = (recon * recon).mean() + lam_t * l0
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, n_steps // 8) == 0 or step == n_steps - 1:
            with torch.no_grad():
                m_eval = _hard_concrete_sample(z, beta_t, gamma, zeta, deterministic=True)
                history.append({'step': step, 'beta': beta_t, 'lambda_l0': lam_t, 'recon_loss': float((recon * recon).mean().detach().item()), 'expected_l0': float(l0.detach().item()), 'n_active_eval': int((m_eval > 0.5).sum().item())})
    with torch.no_grad():
        m_final = _hard_concrete_sample(z, beta_end, gamma, zeta, deterministic=True).detach()
    eap = (G.T @ q).abs()
    score = m_final + 0.001 * eap / (eap.max() + 1e-09)
    sel = torch.topk(score, int(min(k, n_cand))).indices.tolist()
    drivers = [discovery.candidate_nodes[i] for i in sel]
    strengths = ridge_solve(G[:, sel], q, ridge=discovery.ridge)
    _, scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: deltas[n] for n in drivers}, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores, validation_log=[{'action': 'edge_pruning_official_full_validate', 'n_steps': n_steps, 'lambda_l0': lambda_l0, 'mask_active_at_eval': int((m_final > 0.5).sum().item()), 'training_history': history, **scores}], selected_indices=sel, method='edge_pruning_official_full')

def hap_official_baseline(discovery: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], k: int, pre_filter_top: int=64, n_steps: int=800, lambda_l0: float=0.05, lr: float=0.05, n_mc_samples: int=4) -> DriverResult:
    """Faithful HAP baseline: EAP pre-filter + hard-concrete L0 mask training.

    Mirrors Gu et al. 2025's two-stage architecture: stage 1 attribution-based
    pruning, stage 2 differentiable L0 mask refinement on the survivors.
    The improvement over ``hap_proxy_baseline`` is that stage 2 now uses the
    Louizos 2018 hard-concrete distribution with annealed β/λ_L0, matching
    the differentiable-binarization step the official HAP paper relies on.
    """
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    eap_scores = (G.T @ q).abs()
    pool = torch.topk(eap_scores, min(pre_filter_top, G.shape[1])).indices.tolist()
    G_sub = G[:, pool]
    m_sub, _, history = _train_hard_concrete_node_mask(G_sub, q, n_steps=n_steps, lambda_l0_final=lambda_l0, lr=lr, n_mc_samples=n_mc_samples)
    eap_sub = eap_scores[pool]
    score_sub = m_sub + 0.001 * eap_sub / (eap_sub.max() + 1e-09)
    top_k = int(min(k, len(pool)))
    sub_sel = torch.topk(score_sub, top_k).indices.tolist()
    sel = [pool[i] for i in sub_sel]
    drivers = [discovery.candidate_nodes[i] for i in sel]
    strengths = ridge_solve(G[:, sel], q, ridge=discovery.ridge)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    _, scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, {n: deltas[n] for n in drivers}, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores, validation_log=[{'action': 'hap_official_validate', 'pre_filter_top': pre_filter_top, 'n_steps': n_steps, 'lambda_l0': lambda_l0, 'n_mc_samples': n_mc_samples, 'mask_active_at_eval': int((m_sub > 0.5).sum().item()), 'training_history': history, **scores}], selected_indices=sel, method='hap_official')

def endpoint_nodes_from_file(discovery: DriverCircuitDiscovery, path: str | Path, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], method_name: str='edge_pruning') -> DriverResult:
    """Load a pre-computed edge set and convert endpoints to a node set.

    Expected JSON schema:
        {
          "edges": [
            {"u": {"basis": "head_pos", "layer": 5, "position": "END", "component": 5},
             "v": {"basis": "head_pos", "layer": 9, "position": "END", "component": 9}},
            ...
          ]
        }

    The nodes must match candidates registered on ``discovery.candidate_nodes``.
    """
    data = json.loads(Path(path).read_text())
    edges = data['edges']
    registry = {discovery.candidate_nodes[i].label(): i for i in range(len(discovery.candidate_nodes))}
    endpoints: List[int] = []
    for edge in edges:
        for key in ('u', 'v'):
            spec = edge[key]
            node = DriverNode(**spec)
            if node.label() in registry:
                idx = registry[node.label()]
                if idx not in endpoints:
                    endpoints.append(idx)
    drivers = [discovery.candidate_nodes[i] for i in endpoints]
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    strengths = ridge_solve(G[:, endpoints], q, ridge=discovery.ridge) if endpoints else torch.empty(0, device=q.device, dtype=q.dtype)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    _, scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, deltas, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=[], nonlinear_scores=scores, validation_log=[{'action': f'{method_name}_endpoint_validate', **scores}], selected_indices=endpoints, method=method_name)
