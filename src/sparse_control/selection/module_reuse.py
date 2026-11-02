"""
Sample-normalized module-level reuse selection and fitting kernels; separate from raw-dictionary selection.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Protocol, Sequence, Tuple, Union
import math
import torch
from torch import Tensor
PositionSpec = Union[int, str]

@dataclass(frozen=True)
class DriverNode:
    """A candidate internal intervention unit.

    Examples
    --------
    * ``DriverNode("head_pos", layer=9, position="END", component=9)``
    * ``DriverNode("resid_coord", layer=7, position="END", feature=1234)``
    * ``DriverNode("sae_feature", layer=8, position="S2", feature=913)``

    ``position`` may be an ``int`` (fixed token index, negative allowed) or a
    string label. The adapter is responsible for resolving string labels into
    per-sample token indices via ``batch["positions"]``.
    """
    basis: str
    layer: int
    position: PositionSpec
    component: Optional[int] = None
    feature: Optional[int] = None
    name: Optional[str] = None

    def label(self) -> str:
        if self.name:
            return self.name
        parts = [self.basis, f'L{self.layer}', f'P{self.position}']
        if self.component is not None:
            parts.append(f'H{self.component}')
        if self.feature is not None:
            parts.append(f'F{self.feature}')
        return ':'.join(parts)

@dataclass
class DriverResult:
    drivers: List[DriverNode]
    strengths: Tensor
    linear_residual_norms: List[float]
    nonlinear_scores: Dict[str, float]
    validation_log: List[Dict[str, Any]]
    selected_indices: List[int]
    method: str = 'driver_omp'

class TargetMetric(Protocol):
    """Callable task target.

    Returns a tensor of shape ``[batch]`` (scalar target) or ``[batch, d_T]``
    (vector target). Examples: IOI logit-diff is ``[batch]``; MCQA logits over
    {A, B, C, D} are ``[batch, 4]``; a probe direction score is ``[batch]``.
    """

    def __call__(self, model_outputs: Any, batch: Mapping[str, Any]) -> Tensor:
        ...

class HookAdapter(Protocol):
    """Model-specific adapter.

    Implementations must:
      * resolve ``DriverNode.position`` (int or label) against ``batch``;
      * place tensors on ``self.device``;
      * support scalar AND vector-valued targets via ``target_sketch``.

    Shapes returned by ``jacobian_wrt_nodes``:
      * scalar target:        ``[batch, *act_dims]``
      * sketched vector tgt:  ``[batch, sketch_dim, *act_dims]``
    """
    device: str

    def cache_activations(self, batch: Mapping[str, Any], nodes: Sequence[DriverNode]) -> Dict[DriverNode, Tensor]:
        ...

    def jacobian_wrt_nodes(self, batch: Mapping[str, Any], nodes: Sequence[DriverNode], target_metric: TargetMetric, target_sketch: Optional[Tensor]=None) -> Dict[DriverNode, Tensor]:
        ...

    def run_with_patches(self, batch: Mapping[str, Any], patches: Mapping[DriverNode, Tensor], target_metric: TargetMetric) -> Tensor:
        ...

    def run_clean_or_corrupt(self, batch: Mapping[str, Any], mode: str, target_metric: TargetMetric) -> Tensor:
        ...

def flatten_target(x: Tensor) -> Tensor:
    """Flatten ``[batch]`` or ``[batch, d_T]`` into ``[batch * d_T]``."""
    return x.detach().reshape(-1).float()

def ridge_solve(G: Tensor, q: Tensor, ridge: float=1e-05) -> Tensor:
    """Solve ``min_u ||q - G u||^2 + ridge ||u||^2``.

    ``G: [n, k]``, ``q: [n]`` -> ``u: [k]``.
    """
    if G.numel() == 0 or G.shape[1] == 0:
        return torch.empty(0, device=q.device, dtype=q.dtype)
    k = G.shape[1]
    A = G.T @ G + ridge * torch.eye(k, device=G.device, dtype=G.dtype)
    b = G.T @ q
    return torch.linalg.solve(A, b)

def pearson_correlation(x: Tensor, y: Tensor) -> float:
    x = x.detach().flatten().float()
    y = y.detach().flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if denom.item() == 0:
        return float('nan')
    return float((x @ y / denom).item())

def topk_overlap(x: Tensor, y: Tensor, k: int) -> float:
    """Fraction of ``top-k`` indices of |x| that appear in the top-k of |y|."""
    x = x.detach().flatten().float()
    y = y.detach().flatten().float()
    k = min(k, x.numel())
    if k <= 0:
        return float('nan')
    sx = set(torch.topk(x.abs(), k).indices.tolist())
    sy = set(torch.topk(y.abs(), k).indices.tolist())
    return len(sx & sy) / float(k)

def linearization_correlation(predicted: Tensor, true_effects: Tensor) -> float:
    """Pearson correlation for local-effect predictions."""
    return pearson_correlation(predicted, true_effects)

def coverage_score_matrix(G: Tensor, q: Tensor, eps_norm: float=1e-06) -> Tensor:
    """Per-(observation, candidate) coverage score `a_iv` in [0, 1].

    Definition:
        a_iv = clip( sign(q_i) * G[i, v] / (|q_i| + eps_norm),  0,  1 )

    Sign-aligned: only nodes that push q_i in the correct direction earn
    positive coverage. Magnitude-bounded: a single huge jacobian column is
    clipped at 1 (cannot saturate coverage on unrelated samples). Tiny-|q|
    samples get small a_iv (eps-dominated denominator) — combined with the
    weight `w_i = |q_i|` they contribute little to the global objective.
    """
    assert G.ndim == 2 and q.ndim == 1, 'G must be [n, m] and q must be [n]'
    sign_q = torch.sign(q).unsqueeze(1)
    denom = q.abs().clamp_min(eps_norm).unsqueeze(1)
    A = (sign_q * G / denom).clamp(min=0.0, max=1.0)
    return A

def coverage_driver(G: Tensor, q: Tensor, max_drivers: int=20, coverage_target: float=0.95, weight_mode: str='absq', eps_norm: float=1e-06, min_marginal_gain: float=0.0001, cover_mode: str='soft', hard_threshold: float=0.2, signed_split: bool=False) -> Tuple[List[int], List[float], List[float]]:
    """Greedy submodular maximisation of soft coverage.

    F(D) = Σ_i w_i · min(1, Σ_{v ∈ D} a_iv)
    Monotone-submodular ⇒ greedy is (1 - 1/e)-approx of the optimum at any
    fixed budget |D| = k. Selection stops on `max_drivers`, on
    weighted_coverage ≥ `coverage_target`, or when marginal gain falls below
    `min_marginal_gain × total_weight`.

    Returns
    -------
    selected         : list[int]    indices into G's columns
    coverage_history : list[float]  weighted coverage after each pick (in [0, 1])
    marginal_gains   : list[float]  per-pick gain in normalised units
    """
    assert G.ndim == 2 and q.ndim == 1
    A = coverage_score_matrix(G, q, eps_norm=eps_norm)
    n, m_orig = A.shape
    if signed_split:
        sign_neg_q = -torch.sign(q).unsqueeze(1)
        denom = q.abs().clamp_min(eps_norm).unsqueeze(1)
        A_neg = (sign_neg_q * G / denom).clamp(min=0.0, max=1.0)
        A = torch.cat([A, A_neg], dim=1)
    m = A.shape[1]
    if cover_mode == 'hard':
        A = (A >= hard_threshold).to(A.dtype)
    elif cover_mode != 'soft':
        raise ValueError(f'unknown cover_mode={cover_mode!r}')
    if weight_mode == 'absq':
        w = q.abs().clamp_min(eps_norm)
    elif weight_mode == 'uniform':
        w = torch.ones_like(q)
    else:
        raise ValueError(f'unknown weight_mode={weight_mode!r}')
    total_w = float(w.sum().clamp_min(1e-12).item())
    cov = torch.zeros(n, dtype=A.dtype, device=A.device)
    selected: List[int] = []
    coverage_history: List[float] = []
    marginal_gains: List[float] = []
    for _ in range(min(max_drivers, m)):
        new_cov = (cov.unsqueeze(1) + A).clamp_max(1.0)
        gain = ((new_cov - cov.unsqueeze(1)) * w.unsqueeze(1)).sum(dim=0)
        if selected:
            gain[selected] = -float('inf')
        best = int(gain.argmax().item())
        best_gain = float(gain[best].item())
        if not math.isfinite(best_gain) or best_gain <= min_marginal_gain * total_w:
            break
        selected.append(best)
        cov = (cov + A[:, best]).clamp_max(1.0)
        wc = float((cov * w).sum().item()) / total_w
        coverage_history.append(wc)
        marginal_gains.append(best_gain / total_w)
        if wc >= coverage_target:
            break
    if signed_split:
        selected = [i % m_orig for i in selected]
        seen = set()
        deduped = []
        for i in selected:
            if i not in seen:
                seen.add(i)
                deduped.append(i)
        selected = deduped
    return (selected, coverage_history, marginal_gains)

def orthogonal_matching_pursuit(G: Tensor, q: Tensor, costs: Optional[Tensor]=None, max_drivers: int=20, ridge: float=1.0, tol: float=0.05, min_marginal_gain: float=0.001, preselected: Optional[Sequence[int]]=None, H: Optional[Tensor]=None, mu: float=0.0, W_diag: Optional[Tensor]=None, eta_0: float=0.0001) -> Tuple[List[int], Tensor, List[float]]:
    """Driver-OMP (Algorithm C.3) — sparse residual fitting selection.

    Implements the cost-normalised gain rule of Eq. C.27:

        score(v | S_t) = |⟨ ĝ_v, r_t ⟩|² /
                         ((‖ĝ_v‖² + λ W_vv) (c_v + η_0)),

    where the augmented column ``ĝ_v = [Ḡ_v ; √μ Ω^{1/2} H̄_v]`` includes
    the preservation block when ``H`` and ``mu > 0`` are supplied. With the
    defaults ``H=None, mu=0, W_diag=None``, the rule degenerates to plain
    cost-normalised OMP with ``score = |G^T r|² / (‖G_v‖² (c_v + η_0))``.

    Stops when any of:
      * ``max_drivers`` reached,
      * ``‖q − Ĝ_S β‖ / (‖q‖ + ξ) ≤ tol``,
      * marginal residual drop < ``min_marginal_gain`` after at least one pick.

    Defaults: ``ridge=1.0`` and ``tol=0.05`` follow Appendix F.1
    (``λ = 1`` and ``ε_lin = 0.05`` for the linear early-stop).
    """
    assert G.ndim == 2 and q.ndim == 1, 'G must be [n, k] and q must be [n]'
    n_obs, n_cand = G.shape
    if costs is None:
        costs = torch.ones(n_cand, device=G.device, dtype=G.dtype)
    costs = torch.clamp(costs.to(G.device, G.dtype), min=1e-12)
    if H is not None and mu > 0:
        scale = math.sqrt(mu)
        G_hat = torch.cat([G, scale * H], dim=0)
        q_hat = torch.cat([q, q.new_zeros(H.shape[0])], dim=0)
    else:
        G_hat, q_hat = (G, q)
    col_norms_sq = torch.sum(G_hat.pow(2), dim=0).clamp_min(1e-12)
    if W_diag is not None:
        W_diag = torch.clamp(W_diag.to(G.device, G.dtype), min=0.0)
    selected: List[int] = [] if preselected is None else list(preselected)
    residual_norms: List[float] = []
    q_norm = torch.linalg.norm(q_hat).clamp_min(1e-12)
    prev_rel_norm = float('inf')
    for _ in range(max_drivers):
        if selected:
            G_sel = G_hat[:, selected]
            u = ridge_solve(G_sel, q_hat, ridge=ridge)
            residual = q_hat - G_sel @ u
        else:
            residual = q_hat.clone()
        rel_norm = (torch.linalg.norm(residual) / q_norm).item()
        residual_norms.append(rel_norm)
        if rel_norm <= tol:
            break
        if selected and prev_rel_norm - rel_norm < min_marginal_gain:
            break
        prev_rel_norm = rel_norm
        numer = (G_hat.T @ residual).pow(2)
        if W_diag is not None:
            denom_norm = col_norms_sq + ridge * W_diag
        else:
            denom_norm = col_norms_sq
        score = numer / (denom_norm * (costs + eta_0))
        if selected:
            score[selected] = -float('inf')
        best = int(torch.argmax(score).item())
        if not math.isfinite(float(score[best])) or float(score[best]) <= 0:
            break
        selected.append(best)
    strengths = ridge_solve(G_hat[:, selected], q_hat, ridge=ridge) if selected else torch.empty(0, device=q.device, dtype=q.dtype)
    return (selected, strengths, residual_norms)

class DriverCircuitDiscovery:
    """Driver-OMP with nonlinear validation, actual-gain fallback, and pruning."""

    def __init__(self, adapter: HookAdapter, target_metric: TargetMetric, candidate_nodes: Sequence[DriverNode], epsilon: float, max_drivers: int=20, ridge: float=1e-05, lin_tol: float=0.001, min_marginal_gain: float=0.0001, target_sketch_dim: Optional[int]=None, sketch_seed: int=0, nonlinear_validation: bool=True, fallback_mode: Literal['omp_residual', 'actual_gain', 'none']='actual_gain', fallback_topk: int=16, backward_prune: bool=True, max_prune_queries: int=64, alpha_clip: Optional[float]=None) -> None:
        self.adapter = adapter
        self.target_metric = target_metric
        self.candidate_nodes = list(candidate_nodes)
        self.epsilon = float(epsilon)
        self.max_drivers = int(max_drivers)
        self.ridge = float(ridge)
        self.lin_tol = float(lin_tol)
        self.min_marginal_gain = float(min_marginal_gain)
        self.target_sketch_dim = target_sketch_dim
        self.sketch_seed = int(sketch_seed)
        self.do_nonlinear_validation = bool(nonlinear_validation)
        self.fallback_mode = fallback_mode
        self.fallback_topk = int(fallback_topk)
        self.do_backward_prune = bool(backward_prune)
        self.max_prune_queries = int(max_prune_queries)
        self.alpha_clip = float(alpha_clip) if alpha_clip is not None else None

    def _make_sketch(self, target_dim: int) -> Optional[Tensor]:
        if target_dim <= 1:
            return None
        if self.target_sketch_dim is None or target_dim <= self.target_sketch_dim:
            return None
        g = torch.Generator(device='cpu').manual_seed(self.sketch_seed)
        S = torch.randn(target_dim, self.target_sketch_dim, generator=g)
        S = S / math.sqrt(self.target_sketch_dim)
        return S.to(self.adapter.device)

    def estimate_local_effects(self, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any]) -> Tuple[Tensor, Tensor, Dict[DriverNode, Tensor]]:
        clean_targets = self.adapter.run_clean_or_corrupt(clean_batch, 'clean', self.target_metric)
        corrupt_targets = self.adapter.run_clean_or_corrupt(corrupt_batch, 'corrupt', self.target_metric)
        assert clean_targets.shape == corrupt_targets.shape, f'paired batches must yield matching target shapes; got {clean_targets.shape} vs {corrupt_targets.shape}'
        diff = (clean_targets - corrupt_targets).detach()
        if diff.ndim == 1:
            sketch: Optional[Tensor] = None
            q_mat = diff.unsqueeze(-1)
        else:
            target_dim = int(diff.shape[-1])
            sketch = self._make_sketch(target_dim)
            q_mat = diff @ sketch if sketch is not None else diff
        q = q_mat.reshape(-1).float()
        clean_cache = self.adapter.cache_activations(clean_batch, self.candidate_nodes)
        corrupt_cache = self.adapter.cache_activations(corrupt_batch, self.candidate_nodes)
        jac = self.adapter.jacobian_wrt_nodes(corrupt_batch, self.candidate_nodes, self.target_metric, target_sketch=sketch)
        cols: List[Tensor] = []
        deltas: Dict[DriverNode, Tensor] = {}
        for node in self.candidate_nodes:
            delta = (clean_cache[node] - corrupt_cache[node]).detach()
            deltas[node] = delta
            J = jac[node]
            if J.ndim == delta.ndim:
                J = J.unsqueeze(1)
            assert J.shape[0] == delta.shape[0], f'batch mismatch on {node.label()}: jac {J.shape} vs delta {delta.shape}'
            d_prime = int(J.shape[1])
            delta_exp = delta.unsqueeze(1).expand(-1, d_prime, *[-1] * (delta.ndim - 1))
            effect = (J * delta_exp).reshape(J.shape[0], d_prime, -1).sum(dim=-1)
            cols.append(effect.reshape(-1).float())
        G = torch.stack(cols, dim=1)
        return (G, q, deltas)

    def validate_driver_set(self, corrupt_batch: Mapping[str, Any], clean_target: Tensor, drivers: Sequence[DriverNode], deltas: Mapping[DriverNode, Tensor], strengths: Optional[Tensor]=None) -> Tuple[bool, Dict[str, float]]:
        patches: Dict[DriverNode, Tensor] = {}
        clip = self.alpha_clip
        for j, node in enumerate(drivers):
            alpha = float(strengths[j].item()) if strengths is not None and len(strengths) > j else 1.0
            if clip is not None:
                alpha = max(-clip, min(clip, alpha))
            patches[node] = alpha * deltas[node]
        patched_target = self.adapter.run_with_patches(corrupt_batch, patches, self.target_metric)
        clean_flat = flatten_target(clean_target)
        patched_flat = flatten_target(patched_target)
        mae = torch.mean(torch.abs(clean_flat - patched_flat)).item()
        rel_l2 = (torch.linalg.norm(clean_flat - patched_flat) / torch.linalg.norm(clean_flat).clamp_min(1e-12)).item()
        ok = mae <= self.epsilon
        return (ok, {'mae_to_clean': mae, 'rel_l2_to_clean': rel_l2})

    def _actual_gain_greedy_pick(self, corrupt_batch: Mapping[str, Any], clean_target: Tensor, selected: List[int], remaining: List[int], G: Tensor, q: Tensor, deltas: Mapping[DriverNode, Tensor], topk: int) -> Optional[Tuple[int, Dict[str, float]]]:
        if not remaining:
            return None
        if selected:
            u = ridge_solve(G[:, selected], q, ridge=self.ridge)
            residual = q - G[:, selected] @ u
        else:
            residual = q
        scores = torch.abs(G[:, remaining].T @ residual)
        k = min(topk, len(remaining))
        top_idx = torch.topk(scores, k).indices.tolist()
        candidates = [remaining[i] for i in top_idx]
        best_cand: Optional[int] = None
        best_rel = float('inf')
        best_scores: Dict[str, float] = {}
        for cand in candidates:
            trial_sel = selected + [cand]
            trial_drivers = [self.candidate_nodes[i] for i in trial_sel]
            trial_strengths = ridge_solve(G[:, trial_sel], q, ridge=self.ridge)
            _, sv = self.validate_driver_set(corrupt_batch, clean_target, trial_drivers, deltas, trial_strengths)
            if sv['rel_l2_to_clean'] < best_rel:
                best_rel = sv['rel_l2_to_clean']
                best_cand = cand
                best_scores = sv
        if best_cand is None:
            return None
        return (best_cand, best_scores)

    def backward_prune(self, corrupt_batch: Mapping[str, Any], clean_target: Tensor, drivers: List[DriverNode], deltas: Mapping[DriverNode, Tensor], strengths: Tensor) -> Tuple[List[DriverNode], Tensor, List[Dict[str, Any]]]:
        log: List[Dict[str, Any]] = []
        queries = 0
        keep = list(range(len(drivers)))
        removal_order = sorted(keep, key=lambda i: float(strengths[i].abs().item()))
        changed = True
        while changed and queries < self.max_prune_queries:
            changed = False
            for idx in list(removal_order):
                if idx not in keep:
                    continue
                trial_keep = [i for i in keep if i != idx]
                trial_drivers = [drivers[i] for i in trial_keep]
                trial_strengths = torch.stack([strengths[i] for i in trial_keep]) if trial_keep else torch.empty(0, device=strengths.device, dtype=strengths.dtype)
                ok, scores = self.validate_driver_set(corrupt_batch, clean_target, trial_drivers, deltas, trial_strengths)
                queries += 1
                log.append({'action': 'try_remove', 'node': drivers[idx].label(), 'ok': ok, **scores})
                if ok:
                    keep.remove(idx)
                    changed = True
                    break
                if queries >= self.max_prune_queries:
                    break
        pruned_drivers = [drivers[i] for i in keep]
        pruned_strengths = torch.stack([strengths[i] for i in keep]) if keep else torch.empty(0, device=strengths.device, dtype=strengths.dtype)
        return (pruned_drivers, pruned_strengths, log)

    def run(self, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], costs: Optional[Tensor]=None) -> DriverResult:
        G, q, deltas = self.estimate_local_effects(clean_batch, corrupt_batch)
        selected, strengths, residual_norms = orthogonal_matching_pursuit(G, q, costs=costs, max_drivers=self.max_drivers, ridge=self.ridge, tol=self.lin_tol, min_marginal_gain=self.min_marginal_gain)
        drivers = [self.candidate_nodes[i] for i in selected]
        val_log: List[Dict[str, Any]] = []
        if not self.do_nonlinear_validation:
            return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=residual_norms, nonlinear_scores={}, validation_log=val_log, selected_indices=selected, method='driver_omp_linear_only')
        clean_target = self.adapter.run_clean_or_corrupt(clean_batch, 'clean', self.target_metric)
        ok, scores = self.validate_driver_set(corrupt_batch, clean_target, drivers, deltas, strengths)
        val_log.append({'action': 'initial_validate', 'ok': ok, **scores})
        while not ok and self.fallback_mode != 'none' and (len(selected) < self.max_drivers):
            remaining = [i for i in range(len(self.candidate_nodes)) if i not in selected]
            if not remaining:
                break
            if self.fallback_mode == 'actual_gain':
                picked = self._actual_gain_greedy_pick(corrupt_batch, clean_target, selected, remaining, G, q, deltas, topk=self.fallback_topk)
                if picked is None:
                    break
                cand, sv = picked
            else:
                if selected:
                    u = ridge_solve(G[:, selected], q, ridge=self.ridge)
                    residual = q - G[:, selected] @ u
                else:
                    residual = q
                rank = torch.argsort(torch.abs(G[:, remaining].T @ residual), descending=True)
                cand = remaining[int(rank[0].item())]
                sv = {}
            selected.append(cand)
            drivers = [self.candidate_nodes[i] for i in selected]
            strengths = ridge_solve(G[:, selected], q, ridge=self.ridge)
            ok, scores = self.validate_driver_set(corrupt_batch, clean_target, drivers, deltas, strengths)
            val_log.append({'action': f'fallback_{self.fallback_mode}', 'node': self.candidate_nodes[cand].label(), 'ok': ok, **scores})
        if self.do_backward_prune and drivers:
            drivers, strengths, prune_log = self.backward_prune(corrupt_batch, clean_target, drivers, deltas, strengths)
            val_log.extend(prune_log)
        ok, final_scores = self.validate_driver_set(corrupt_batch, clean_target, drivers, deltas, strengths)
        val_log.append({'action': 'final_validate', 'ok': ok, **final_scores})
        return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=residual_norms, nonlinear_scores=final_scores, validation_log=val_log, selected_indices=selected, method='driver_omp')

def top_node_attribution(G: Tensor, q: Tensor, k: int) -> List[int]:
    """Top-k candidates by absolute linear single-node effect ``|G_v^T q|``."""
    assert G.ndim == 2 and q.ndim == 1
    scores = torch.abs(G.T @ q)
    k = min(k, G.shape[1])
    return torch.topk(scores, k).indices.tolist()

def actual_gain_greedy(discovery: 'DriverCircuitDiscovery', clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], max_drivers: int, topk_proposal: int=32) -> DriverResult:
    """Pure actual-gain greedy baseline."""
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    clean_target = discovery.adapter.run_clean_or_corrupt(clean_batch, 'clean', discovery.target_metric)
    selected: List[int] = []
    val_log: List[Dict[str, Any]] = []
    residual_norms: List[float] = []
    strengths = torch.empty(0, device=q.device, dtype=q.dtype)
    for _ in range(max_drivers):
        remaining = [i for i in range(len(discovery.candidate_nodes)) if i not in selected]
        picked = discovery._actual_gain_greedy_pick(corrupt_batch, clean_target, selected, remaining, G, q, deltas, topk=topk_proposal)
        if picked is None:
            break
        cand, sv = picked
        selected.append(cand)
        strengths = ridge_solve(G[:, selected], q, ridge=discovery.ridge)
        val_log.append({'action': 'actual_gain_step', 'node': discovery.candidate_nodes[cand].label(), **sv})
        residual_norms.append(float(sv.get('rel_l2_to_clean', float('nan'))))
        if sv.get('mae_to_clean', float('inf')) <= discovery.epsilon:
            break
    drivers = [discovery.candidate_nodes[i] for i in selected]
    _, final_scores = discovery.validate_driver_set(corrupt_batch, clean_target, drivers, deltas, strengths)
    return DriverResult(drivers=drivers, strengths=strengths.detach(), linear_residual_norms=residual_norms, nonlinear_scores=final_scores, validation_log=val_log, selected_indices=selected, method='actual_gain_greedy')

def consensus_core(results: Sequence[DriverResult], gamma: float=0.7) -> Tuple[List[DriverNode], List[DriverNode], Dict[DriverNode, float]]:
    """Compute core and backup drivers from repeated runs."""
    counts: Dict[DriverNode, int] = {}
    for res in results:
        for node in set(res.drivers):
            counts[node] = counts.get(node, 0) + 1
    freq = {node: c / max(1, len(results)) for node, c in counts.items()}
    core = [node for node, p in freq.items() if p >= gamma]
    backup = [node for node, p in freq.items() if 0 < p < gamma]
    return (core, backup, freq)

def build_local_tables(discovery: 'DriverCircuitDiscovery', clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any], off_target_batch: Optional[Mapping[str, Any]]=None, preservation_target: Optional['TargetMetric']=None) -> Dict[str, Any]:
    G, q, deltas = discovery.estimate_local_effects(clean_batch, corrupt_batch)
    n_target = max(1, q.shape[0])
    q_bar = q / math.sqrt(n_target)
    G_bar = G / math.sqrt(n_target)
    nodes = discovery.candidate_nodes
    M = len(nodes)
    diag = torch.zeros(M, device=G_bar.device, dtype=G_bar.dtype)
    for j, node in enumerate(nodes):
        u = deltas[node]
        flat = u.reshape(u.shape[0], -1).float()
        diag[j] = (flat * flat).sum(dim=-1).mean()
    W = torch.diag(diag)
    H_bar: Optional[Tensor] = None
    if off_target_batch is not None:
        prev_metric = discovery.target_metric
        try:
            if preservation_target is not None:
                discovery.target_metric = preservation_target
            H_raw, _q_off, _d_off = discovery.estimate_local_effects(off_target_batch, off_target_batch)
        finally:
            discovery.target_metric = prev_metric
        n_off = max(1, H_raw.shape[0])
        H_bar = H_raw / math.sqrt(n_off)
    return {'q': q_bar, 'G': G_bar, 'H': H_bar, 'W': W, 'deltas': deltas, 'candidate_nodes': list(nodes)}
driver_omp = orthogonal_matching_pursuit
'Paper alias for Algorithm C.3 (Driver-OMP).'


def fit_reuse_coefficients(G, q, selected, ridge=1.0):
    """Fit coefficients after the module-series sqrt(nrow) normalization."""
    scale = math.sqrt(q.numel())
    return ridge_solve(G[:, selected] / scale, q / scale, ridge=ridge)

def select_support(method, G_normalized, q_normalized, nodes, budget, seed, ridge=1.0):
    from ..runners.implementations.module_reuse import _pick
    return _pick(method, G_normalized, q_normalized, nodes, budget, seed, ridge)
