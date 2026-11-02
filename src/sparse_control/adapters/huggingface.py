"""
Hugging Face residual, attention-output and MLP-output adapter.
"""
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import torch
from torch import Tensor, nn
from sparse_control.selection.core import DriverNode, TargetMetric
from sparse_control.adapters.transformerlens import _gather_at_positions, _scatter_add_at_positions, _resolve_position, _expand_batch_for_sketch
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
SubmodulePath = str

class HFHookAdapter:
    """Driver-Circuits adapter for HuggingFace causal LMs."""

    def __init__(self, model: nn.Module, tokenizer: Any, device: Optional[str]=None, layers_attr: str='model.layers', resid_module: Callable[[int], SubmodulePath]=lambda l: f'model.layers.{l}', mlp_module: Callable[[int], SubmodulePath]=lambda l: f'model.layers.{l}.mlp', attn_module: Callable[[int], SubmodulePath]=lambda l: f'model.layers.{l}.self_attn') -> None:
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        self.layers_attr = layers_attr
        self.resid_module = resid_module
        self.mlp_module = mlp_module
        self.attn_module = attn_module

    def _submodule(self, path: SubmodulePath) -> nn.Module:
        mod: Any = self.model
        for part in path.split('.'):
            mod = getattr(mod, part)
        return mod

    def _module_path(self, node: DriverNode) -> SubmodulePath:
        if node.basis in {'resid_post', 'resid_coord'}:
            return self.resid_module(node.layer)
        if node.basis == 'mlp_out':
            return self.mlp_module(node.layer)
        if node.basis == 'attn_out':
            return self.attn_module(node.layer)
        raise ValueError(f'Unsupported basis for HF adapter: {node.basis}')

    @contextmanager
    def _forward_hooks(self, save: Dict[SubmodulePath, Tensor], paths: Sequence[SubmodulePath]):
        """Register forward hooks that stash the full activation at each path."""
        handles = []

        def make_hook(path: SubmodulePath):

            def hook(_module: nn.Module, _inputs: Any, output: Any):
                act = output[0] if isinstance(output, tuple) else output
                save[path] = act
                return output
            return hook
        try:
            for p in paths:
                handles.append(self._submodule(p).register_forward_hook(make_hook(p)))
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def _patch_hooks(self, patches_by_path: Mapping[SubmodulePath, List[Tuple[DriverNode, Tensor]]], batch: Mapping[str, Any]):
        handles = []

        def make_hook(path: SubmodulePath):
            specs = patches_by_path[path]

            def hook(_module: nn.Module, _inputs: Any, output: Any):
                if isinstance(output, tuple):
                    act = output[0]
                    rest = output[1:]
                else:
                    act = output
                    rest = None
                out = act
                for node, delta in specs:
                    out = self._scatter_into_full(out, node, delta, alpha=1.0, batch=batch)
                return (out, *rest) if rest is not None else out
            return hook
        try:
            for p in patches_by_path:
                handles.append(self._submodule(p).register_forward_hook(make_hook(p)))
            yield
        finally:
            for h in handles:
                h.remove()

    def _select_from_full(self, full_act: Tensor, node: DriverNode, batch: Mapping[str, Any]) -> Tensor:
        pos = _resolve_position(node, batch)
        sliced = _gather_at_positions(full_act, pos)
        if node.basis in {'resid_post', 'mlp_out', 'attn_out'}:
            return sliced
        if node.basis == 'resid_coord':
            if node.feature is None:
                raise ValueError('resid_coord node must specify feature (coord index)')
            return sliced[:, int(node.feature)].unsqueeze(-1)
        raise ValueError(node.basis)

    def _scatter_into_full(self, full_act: Tensor, node: DriverNode, delta: Tensor, alpha: float, batch: Mapping[str, Any]) -> Tensor:
        pos = _resolve_position(node, batch)
        if node.basis in {'resid_post', 'mlp_out', 'attn_out'}:
            return _scatter_add_at_positions(full_act, pos, (alpha * delta).to(full_act.device, full_act.dtype))
        if node.basis == 'resid_coord':
            add = torch.zeros_like(_gather_at_positions(full_act, pos))
            add[:, int(node.feature)] = alpha * delta.squeeze(-1).to(add.device, add.dtype)
            return _scatter_add_at_positions(full_act, pos, add)
        raise ValueError(node.basis)

    def _model_kwargs(self, batch: Mapping[str, Any]) -> Dict[str, Tensor]:
        tokens = batch['tokens'].to(self.device)
        kwargs: Dict[str, Tensor] = {'input_ids': tokens}
        attn_mask = batch.get('attention_mask')
        if attn_mask is not None:
            kwargs['attention_mask'] = attn_mask.to(self.device)
        return kwargs

    def cache_activations(self, batch: Mapping[str, Any], nodes: Sequence[DriverNode]) -> Dict[DriverNode, Tensor]:
        paths = sorted({self._module_path(n) for n in nodes})
        saved: Dict[SubmodulePath, Tensor] = {}
        with torch.no_grad():
            with self._forward_hooks(saved, paths):
                self.model(**self._model_kwargs(batch))
        out: Dict[DriverNode, Tensor] = {}
        for n in nodes:
            out[n] = self._select_from_full(saved[self._module_path(n)], n, batch).detach()
        return out

    def run_clean_or_corrupt(self, batch: Mapping[str, Any], mode: str, target_metric: TargetMetric) -> Tensor:
        with torch.no_grad():
            out = self.model(**self._model_kwargs(batch))
            logits = out.logits if hasattr(out, 'logits') else out[0]
            return target_metric(logits, batch).detach()

    def jacobian_wrt_nodes(self, batch: Mapping[str, Any], nodes: Sequence[DriverNode], target_metric: TargetMetric, target_sketch: Optional[Tensor]=None) -> Dict[DriverNode, Tensor]:
        paths = sorted({self._module_path(n) for n in nodes})
        saved: Dict[SubmodulePath, Tensor] = {}
        with self._forward_hooks(saved, paths):
            out = self.model(**self._model_kwargs(batch))
            logits = out.logits if hasattr(out, 'logits') else out[0]
            target = target_metric(logits, batch)
        if target.ndim == 1:
            grad_outputs = torch.ones_like(target)
            grads_full = torch.autograd.grad(target, list(saved.values()), grad_outputs=grad_outputs, retain_graph=False, allow_unused=False)
            name_to_grad = dict(zip(saved.keys(), grads_full))
            out_dict: Dict[DriverNode, Tensor] = {}
            for n in nodes:
                full_grad = name_to_grad[self._module_path(n)]
                out_dict[n] = self._select_from_full(full_grad, n, batch).detach()
            return out_dict
        sketched = target @ target_sketch if target_sketch is not None else target
        d_prime = int(sketched.shape[1])
        name_to_jac: Dict[SubmodulePath, List[Tensor]] = {p: [] for p in saved}
        for j in range(d_prime):
            grad_outputs = torch.zeros_like(sketched)
            grad_outputs[:, j] = 1.0
            grads_full = torch.autograd.grad(sketched, list(saved.values()), grad_outputs=grad_outputs, retain_graph=j < d_prime - 1, allow_unused=False)
            for name, g in zip(saved.keys(), grads_full):
                name_to_jac[name].append(g.detach())
        result: Dict[DriverNode, Tensor] = {}
        for n in nodes:
            stacked = torch.stack(name_to_jac[self._module_path(n)], dim=1)
            b = stacked.shape[0]
            dp = stacked.shape[1]
            merged = stacked.reshape(b * dp, *stacked.shape[2:])
            expanded_batch = _expand_batch_for_sketch(batch, dp)
            selected = self._select_from_full(merged, n, expanded_batch)
            result[n] = selected.reshape(b, dp, *selected.shape[1:])
        return result

    def run_with_patches(self, batch: Mapping[str, Any], patches: Mapping[DriverNode, Tensor], target_metric: TargetMetric) -> Tensor:
        patches_by_path: Dict[SubmodulePath, List[Tuple[DriverNode, Tensor]]] = {}
        for node, delta in patches.items():
            patches_by_path.setdefault(self._module_path(node), []).append((node, delta))
        with torch.no_grad():
            with self._patch_hooks(patches_by_path, batch):
                out = self.model(**self._model_kwargs(batch))
                logits = out.logits if hasattr(out, 'logits') else out[0]
                return target_metric(logits, batch).detach()
