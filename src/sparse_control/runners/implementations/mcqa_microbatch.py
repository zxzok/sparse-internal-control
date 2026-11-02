"""
Order-preserving microbatch scheduling for MCQA local effect estimation.
"""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping
from typing import Any
import torch
from sparse_control.selection.core import DriverCircuitDiscovery
_ORIGINAL_ESTIMATE = DriverCircuitDiscovery.estimate_local_effects

def _slice_batch(batch: Mapping[str, Any], indices: torch.Tensor, full_batch_size: int) -> dict[str, Any]:
    """Slice only fields whose first dimension is the example dimension."""
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if key == 'positions' and isinstance(value, Mapping):
            sliced[key] = {position_key: position_value.index_select(0, indices) if torch.is_tensor(position_value) and position_value.ndim >= 1 and (int(position_value.shape[0]) == full_batch_size) else position_value for position_key, position_value in value.items()}
        elif torch.is_tensor(value) and value.ndim >= 1 and (int(value.shape[0]) == full_batch_size):
            sliced[key] = value.index_select(0, indices)
        else:
            sliced[key] = value
    return sliced

def install_microbatched_estimator(microbatch_size: int) -> None:
    if microbatch_size <= 0:
        raise ValueError('microbatch_size must be positive')

    def estimate_local_effects_microbatched(self: DriverCircuitDiscovery, clean_batch: Mapping[str, Any], corrupt_batch: Mapping[str, Any]):
        clean_tokens = clean_batch['tokens']
        corrupt_tokens = corrupt_batch['tokens']
        batch_size = int(clean_tokens.shape[0])
        if int(corrupt_tokens.shape[0]) != batch_size:
            raise ValueError('clean and corrupt batch sizes differ')
        if batch_size <= microbatch_size:
            return _ORIGINAL_ESTIMATE(self, clean_batch, corrupt_batch)
        g_chunks = []
        q_chunks = []
        delta_chunks: dict[Any, list[torch.Tensor]] = defaultdict(list)
        device = clean_tokens.device
        for start in range(0, batch_size, microbatch_size):
            stop = min(start + microbatch_size, batch_size)
            indices = torch.arange(start, stop, device=device)
            clean_slice = _slice_batch(clean_batch, indices, batch_size)
            corrupt_slice = _slice_batch(corrupt_batch, indices, batch_size)
            g_chunk, q_chunk, deltas_chunk = _ORIGINAL_ESTIMATE(self, clean_slice, corrupt_slice)
            g_chunks.append(g_chunk.detach())
            q_chunks.append(q_chunk.detach())
            for node, value in deltas_chunk.items():
                delta_chunks[node].append(value.detach())
            if torch.cuda.is_available() and str(device).startswith('cuda'):
                torch.cuda.empty_cache()
        ordered_deltas = {node: torch.cat(parts, dim=0) for node, parts in delta_chunks.items()}
        return (torch.cat(g_chunks, dim=0), torch.cat(q_chunks, dim=0), ordered_deltas)
    DriverCircuitDiscovery.estimate_local_effects = estimate_local_effects_microbatched
