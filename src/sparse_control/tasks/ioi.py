"""Paired IOI input and readout interfaces."""
from .pairs import build_ioi_pairs
from ..adapters.transformerlens import IOILogitDiff
def ioi_margin(logits, io_ids, subject_ids):
    import torch
    i=torch.arange(logits.shape[0], device=logits.device)
    return logits[i, io_ids]-logits[i, subject_ids]
