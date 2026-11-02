"""
Top-k, matched random, nonlinear greedy and Coverage-Driver comparison methods.
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
_WANG_2023_IOI_HEADS = [(2, 2), (4, 11), (0, 1), (0, 10), (3, 0), (5, 5), (5, 8), (5, 9), (6, 9), (7, 3), (7, 9), (8, 6), (8, 10), (9, 6), (9, 9), (10, 0), (10, 10), (11, 2), (10, 7), (11, 10)]
