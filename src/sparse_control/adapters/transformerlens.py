"""
TransformerLens hook_z and module activation adapter and task readouts.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import torch
from torch import Tensor
from sparse_control.selection.core import DriverNode, TargetMetric
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

def _resolve_position(node: DriverNode, batch: Mapping[str, Any]) -> Tensor:
    """Return a ``LongTensor[batch]`` of token positions for this node.

    ``int`` positions are broadcast. String labels are looked up in
    ``batch["positions"]``.
    """
    seq_len_ref = batch.get('tokens')
    batch_size = int(seq_len_ref.shape[0]) if torch.is_tensor(seq_len_ref) else 1
    device = seq_len_ref.device if torch.is_tensor(seq_len_ref) else 'cpu'
    if isinstance(node.position, int):
        pos = int(node.position)
        if pos < 0 and torch.is_tensor(seq_len_ref):
            pos = int(seq_len_ref.shape[1]) + pos
        return torch.full((batch_size,), pos, dtype=torch.long, device=device)
    if isinstance(node.position, str):
        positions_map = batch.get('positions')
        if positions_map is None or node.position not in positions_map:
            raise KeyError(f"Node requires position label {node.position!r}, but batch['positions'] does not provide it. Available: {(list(positions_map.keys()) if positions_map else [])}")
        pos_tensor = positions_map[node.position]
        if not torch.is_tensor(pos_tensor):
            pos_tensor = torch.as_tensor(pos_tensor, dtype=torch.long)
        return pos_tensor.to(device=device, dtype=torch.long)
    raise TypeError(f'Unsupported position spec: {node.position!r}')

def _gather_at_positions(act: Tensor, pos: Tensor) -> Tensor:
    """Gather ``act[b, pos[b], ...]`` along the sequence dimension.

    Supports:
      * ``[batch, seq, d]``       -> ``[batch, d]``
      * ``[batch, seq, h, d]``    -> ``[batch, h, d]``
    """
    idx = pos.view(-1, 1, *[1] * (act.ndim - 2)).expand(pos.shape[0], 1, *act.shape[2:])
    return act.gather(1, idx).squeeze(1)

def _scatter_add_at_positions(act: Tensor, pos: Tensor, add_val: Tensor) -> Tensor:
    """Return ``act`` with ``add_val`` added at ``act[b, pos[b], ...]``.

    Non-in-place so that the autograd graph upstream remains intact.
    """
    out = act.clone()
    batch_idx = torch.arange(act.shape[0], device=act.device)
    if act.ndim == 3:
        out[batch_idx, pos] = out[batch_idx, pos] + add_val
    elif act.ndim == 4:
        out[batch_idx, pos] = out[batch_idx, pos] + add_val
    else:
        raise ValueError(f'Unsupported activation ndim {act.ndim}')
    return out

class TransformerLensAdapter:
    """Driver-Circuits adapter for TransformerLens HookedTransformer models."""

    def __init__(self, model: Any, device: Optional[str]=None) -> None:
        self.model = model
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device

    def hook_name(self, node: DriverNode) -> str:
        if node.basis == 'head_pos':
            return f'blocks.{node.layer}.attn.hook_z'
        if node.basis == 'mlp_pos':
            return f'blocks.{node.layer}.hook_mlp_out'
        if node.basis in {'resid_pos', 'resid_coord'}:
            return f'blocks.{node.layer}.hook_resid_post'
        raise ValueError(f'Unknown node basis: {node.basis}')

    def _select_from_full(self, full_act: Tensor, node: DriverNode, batch: Mapping[str, Any]) -> Tensor:
        """Given ``[batch, seq, ...]`` cached activation, return the node's slice."""
        pos = _resolve_position(node, batch)
        if node.basis == 'head_pos':
            head_slice = _gather_at_positions(full_act, pos)
            if node.component is None:
                raise ValueError(f'head_pos node must specify component (head index): {node}')
            return head_slice[:, int(node.component), :]
        if node.basis in {'mlp_pos', 'resid_pos'}:
            return _gather_at_positions(full_act, pos)
        if node.basis == 'resid_coord':
            vec = _gather_at_positions(full_act, pos)
            if node.feature is None:
                raise ValueError('resid_coord node must specify feature (coord index)')
            return vec[:, int(node.feature)].unsqueeze(-1)
        raise ValueError(node.basis)

    def _scatter_into_full(self, full_act: Tensor, node: DriverNode, delta: Tensor, alpha: float, batch: Mapping[str, Any]) -> Tensor:
        pos = _resolve_position(node, batch)
        if node.basis == 'head_pos':
            add = torch.zeros_like(_gather_at_positions(full_act, pos))
            add[:, int(node.component), :] = alpha * delta.to(add.device, add.dtype)
            return _scatter_add_at_positions(full_act, pos, add)
        if node.basis in {'mlp_pos', 'resid_pos'}:
            return _scatter_add_at_positions(full_act, pos, (alpha * delta).to(full_act.device, full_act.dtype))
        if node.basis == 'resid_coord':
            add = torch.zeros_like(_gather_at_positions(full_act, pos))
            add[:, int(node.feature)] = alpha * delta.squeeze(-1).to(add.device, add.dtype)
            return _scatter_add_at_positions(full_act, pos, add)
        raise ValueError(node.basis)

    def cache_activations(self, batch: Mapping[str, Any], nodes: Sequence[DriverNode]) -> Dict[DriverNode, Tensor]:
        tokens = batch['tokens'].to(self.device)
        hook_names = list({self.hook_name(n) for n in nodes})
        _, cache = self.model.run_with_cache(tokens, names_filter=lambda name: name in hook_names)
        out: Dict[DriverNode, Tensor] = {}
        for n in nodes:
            out[n] = self._select_from_full(cache[self.hook_name(n)], n, batch).detach()
        return out

    def run_clean_or_corrupt(self, batch: Mapping[str, Any], mode: str, target_metric: TargetMetric) -> Tensor:
        tokens = batch['tokens'].to(self.device)
        with torch.no_grad():
            logits = self.model(tokens)
            return target_metric(logits, batch).detach()

    def jacobian_wrt_nodes(self, batch: Mapping[str, Any], nodes: Sequence[DriverNode], target_metric: TargetMetric, target_sketch: Optional[Tensor]=None) -> Dict[DriverNode, Tensor]:
        """Jacobian of target (optionally sketched) w.r.t. node activations.

        Returns, per node:
          * scalar target:        ``[batch, *act_dims]``
          * sketched vector tgt:  ``[batch, d', *act_dims]``
        """
        tokens = batch['tokens'].to(self.device)
        hook_names = list({self.hook_name(n) for n in nodes})
        saved: Dict[str, Tensor] = {}

        def save_hook(act: Tensor, hook: Any) -> Tensor:
            saved[hook.name] = act
            return act
        hooks = [(name, save_hook) for name in hook_names]
        logits = self.model.run_with_hooks(tokens, fwd_hooks=hooks)
        target = target_metric(logits, batch)
        if target.ndim == 1:
            grad_outputs = torch.ones_like(target)
            grads_full = torch.autograd.grad(target, list(saved.values()), grad_outputs=grad_outputs, retain_graph=False, allow_unused=False)
            name_to_grad = dict(zip(saved.keys(), grads_full))
            out: Dict[DriverNode, Tensor] = {}
            for n in nodes:
                full_grad = name_to_grad[self.hook_name(n)]
                out[n] = self._select_from_full(full_grad, n, batch).detach()
            return out
        if target_sketch is not None:
            sketched = target @ target_sketch
        else:
            sketched = target
        d_prime = int(sketched.shape[1])
        name_to_jac: Dict[str, List[Tensor]] = {name: [] for name in saved}
        for j in range(d_prime):
            grad_outputs = torch.zeros_like(sketched)
            grad_outputs[:, j] = 1.0
            grads_full = torch.autograd.grad(sketched, list(saved.values()), grad_outputs=grad_outputs, retain_graph=j < d_prime - 1, allow_unused=False)
            for name, g in zip(saved.keys(), grads_full):
                name_to_jac[name].append(g.detach())
        out = {}
        for n in nodes:
            stacked = torch.stack(name_to_jac[self.hook_name(n)], dim=1)
            b = stacked.shape[0]
            dp = stacked.shape[1]
            merged = stacked.reshape(b * dp, *stacked.shape[2:])
            expanded_batch = _expand_batch_for_sketch(batch, dp)
            selected = self._select_from_full(merged, n, expanded_batch)
            out[n] = selected.reshape(b, dp, *selected.shape[1:])
        return out

    def run_with_patches(self, batch: Mapping[str, Any], patches: Mapping[DriverNode, Tensor], target_metric: TargetMetric) -> Tensor:
        tokens = batch['tokens'].to(self.device)
        patches_by_hook: Dict[str, List[Tuple[DriverNode, Tensor]]] = {}
        for node, delta in patches.items():
            patches_by_hook.setdefault(self.hook_name(node), []).append((node, delta))

        def make_hook(name: str):

            def patch_hook(act: Tensor, hook: Any) -> Tensor:
                out = act
                for node, delta in patches_by_hook[name]:
                    out = self._scatter_into_full(out, node, delta, alpha=1.0, batch=batch)
                return out
            return patch_hook
        hooks = [(name, make_hook(name)) for name in patches_by_hook]
        with torch.no_grad():
            logits = self.model.run_with_hooks(tokens, fwd_hooks=hooks)
            return target_metric(logits, batch).detach()

def _expand_batch_for_sketch(batch: Mapping[str, Any], d_prime: int) -> Mapping[str, Any]:
    """Repeat per-sample fields used for position resolution along the sketch axis."""
    expanded: Dict[str, Any] = {}
    for k, v in batch.items():
        if k == 'positions' and isinstance(v, dict):
            expanded[k] = {lab: _repeat_per_sample(t, d_prime) if torch.is_tensor(t) else t for lab, t in v.items()}
        elif torch.is_tensor(v) and v.ndim >= 1:
            expanded[k] = _repeat_per_sample(v, d_prime)
        else:
            expanded[k] = v
    return expanded

def _repeat_per_sample(t: Tensor, d_prime: int) -> Tensor:
    """Repeat each sample ``d_prime`` times along batch dim in contiguous order
    so that indexing ``[b * d_prime + j]`` recovers sample ``b`` for any ``j``.
    """
    b = t.shape[0]
    repeated = t.unsqueeze(1).expand(b, d_prime, *t.shape[1:]).reshape(b * d_prime, *t.shape[1:])
    return repeated

class IOILogitDiff:
    """IO logit minus S logit at the END token position.

    ``batch`` requires ``io_token_ids``, ``s_token_ids``, and either
    ``answer_pos`` (int or ``[batch]`` tensor) or ``positions["END"]``.
    """

    def __call__(self, logits: Tensor, batch: Mapping[str, Any]) -> Tensor:
        io = batch['io_token_ids'].to(logits.device)
        s = batch['s_token_ids'].to(logits.device)
        if 'answer_pos' in batch:
            pos = batch['answer_pos']
        elif 'positions' in batch and 'END' in batch['positions']:
            pos = batch['positions']['END']
        else:
            pos = -1
        if torch.is_tensor(pos):
            idx = torch.arange(logits.shape[0], device=logits.device)
            answer_logits = logits[idx, pos.to(logits.device), :]
        else:
            answer_logits = logits[:, int(pos), :]
        io_logit = answer_logits.gather(-1, io[:, None]).squeeze(-1)
        s_logit = answer_logits.gather(-1, s[:, None]).squeeze(-1)
        return io_logit - s_logit

class MCQALogitVector:
    """Return logits over the candidate answer tokens at the answer position.

    ``batch`` requires ``choice_token_ids`` of shape ``[batch, n_choices]`` and
    ``answer_pos`` (int or ``[batch]``) — or ``positions["ANSWER"]``.
    """

    def __call__(self, logits: Tensor, batch: Mapping[str, Any]) -> Tensor:
        choice_ids = batch['choice_token_ids'].to(logits.device)
        if 'answer_pos' in batch:
            pos = batch['answer_pos']
        else:
            pos = batch['positions']['ANSWER']
        if torch.is_tensor(pos):
            idx = torch.arange(logits.shape[0], device=logits.device)
            answer_logits = logits[idx, pos.to(logits.device), :]
        else:
            answer_logits = logits[:, int(pos), :]
        return answer_logits.gather(-1, choice_ids)

class RefusalLogitDiff:
    """Scalar target: log P(refusal) - log P(compliance) at the first
    generation position.

    ``batch`` requires ``refusal_token_ids`` and ``compliance_token_ids``,
    each of shape ``[batch]`` or ``[batch, n_tokens]`` (averaged).
    """

    def __call__(self, logits: Tensor, batch: Mapping[str, Any]) -> Tensor:
        if 'answer_pos' in batch:
            pos = batch['answer_pos']
        else:
            pos = batch['positions']['ANSWER']
        if torch.is_tensor(pos):
            idx = torch.arange(logits.shape[0], device=logits.device)
            ans = logits[idx, pos.to(logits.device), :]
        else:
            ans = logits[:, int(pos), :]
        log_probs = torch.log_softmax(ans, dim=-1)

        def _avg(tok_ids: Tensor) -> Tensor:
            tok_ids = tok_ids.to(log_probs.device)
            if tok_ids.ndim == 1:
                return log_probs.gather(-1, tok_ids[:, None]).squeeze(-1)
            return log_probs.gather(-1, tok_ids).mean(dim=-1)
        return _avg(batch['refusal_token_ids']) - _avg(batch['compliance_token_ids'])
