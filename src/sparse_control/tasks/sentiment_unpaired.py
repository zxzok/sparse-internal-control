"""
Unpaired Sentiment class pools, split helpers and local-effect estimation.
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Tuple
import torch
from sparse_control.selection.core import DriverNode, ridge_solve, orthogonal_matching_pursuit, coverage_driver
from sparse_control.adapters.transformerlens import TransformerLensAdapter
_POS_PROMPTS = ['I just watched this movie and it was absolutely', 'The food at this restaurant is incredibly', 'She gave the best performance ever, truly', 'What a fantastic experience, I felt so', 'The team played brilliantly and ended up', 'After reading the book I was deeply', 'Their customer service is consistently', 'The whole event went smoothly and was', 'He won the award and was understandably', 'The new album is so well-produced and', 'She nailed the interview and emerged', 'Every dish at the dinner was simply', 'The view from the hotel balcony was', 'Her speech moved the audience and felt', 'The product worked exactly as promised, truly', 'The hotel staff were warm and attentive,']
_NEG_PROMPTS = ['I just watched this movie and it was utterly', 'The food at this restaurant is incredibly', 'She gave the worst performance ever, truly', 'What a dreadful experience, I felt so', 'The team played terribly and ended up', 'After reading the book I was deeply', 'Their customer service is consistently', 'The whole event went poorly and was', 'He lost the case and was understandably', 'The new album is so badly-produced and', 'She bombed the interview and emerged', 'Every dish at the dinner was simply', 'The view from the hotel balcony was', 'Her speech bored the audience and felt', 'The product failed exactly as feared, truly', 'The hotel staff were rude and dismissive,']
_POS_WORDS = ['good', 'great', 'wonderful', 'amazing', 'happy', 'excellent', 'fantastic', 'lovely']
_NEG_WORDS = ['bad', 'terrible', 'awful', 'horrible', 'miserable', 'disappointing', 'boring', 'depressing']

class SentimentLogitDiff:

    def __init__(self, pos_token_ids, neg_token_ids):
        self.pos = pos_token_ids
        self.neg = neg_token_ids

    def __call__(self, logits, batch):
        pos = self.pos.to(logits.device)
        neg = self.neg.to(logits.device)
        pos_pos = batch['answer_pos'].to(logits.device) if 'answer_pos' in batch else batch['positions']['ANSWER'].to(logits.device)
        idx = torch.arange(logits.shape[0], device=logits.device)
        ans = logits[idx, pos_pos, :]
        log_p = torch.log_softmax(ans, dim=-1)
        return log_p[:, pos].mean(dim=-1) - log_p[:, neg].mean(dim=-1)

def _build_batch(tokenizer, prompts, device):
    enc = tokenizer(prompts, return_tensors='pt', padding=True)
    last = (enc['attention_mask'].sum(dim=-1) - 1).to(torch.long).to(device)
    return {'tokens': enc['input_ids'].to(device), 'attention_mask': enc['attention_mask'].to(device), 'answer_pos': last, 'positions': {'ANSWER': last}}

def _slice_batch(b: Mapping[str, Any], idxs: torch.Tensor) -> Dict[str, Any]:
    return {'tokens': b['tokens'][idxs], 'attention_mask': b['attention_mask'][idxs], 'answer_pos': b['answer_pos'][idxs], 'positions': {'ANSWER': b['positions']['ANSWER'][idxs]}}

def _broadcast_mean(delta: torch.Tensor, batch_size: int) -> torch.Tensor:
    return delta.mean(dim=0).unsqueeze(0).expand(batch_size, *delta.mean(dim=0).shape)

def estimate_unpaired_local_effects(adapter: TransformerLensAdapter, target, candidate_nodes: List[DriverNode], pos_train: Mapping[str, Any], neg_train: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[DriverNode, torch.Tensor]]:
    """Build (G, q, mean_directions) using only unpaired population means.

    * q_i  = b̄(X⁺_train) - b(x_i^neg_train)   (per-sample, pos-side averaged)
    * Δ̄z_v = z̄_v(X⁺_train) - z̄_v(X⁻_train)    (single direction, broadcast)
    * G[i, v] = J_v(x_i^neg_train) · Δ̄z_v

    No paired counterfactual is used. The X⁺ and X⁻ sets are treated as two
    independent piles whose marginal means are the only driver-selection signal.
    """
    b_pos = adapter.run_clean_or_corrupt(pos_train, 'clean', target).detach().mean()
    b_neg_per = adapter.run_clean_or_corrupt(neg_train, 'corrupt', target).detach()
    q = (b_pos - b_neg_per).reshape(-1).float()
    pos_acts = adapter.cache_activations(pos_train, candidate_nodes)
    neg_acts = adapter.cache_activations(neg_train, candidate_nodes)
    n_neg = neg_train['tokens'].shape[0]
    mean_dirs: Dict[DriverNode, torch.Tensor] = {}
    delta_per_sample: Dict[DriverNode, torch.Tensor] = {}
    for node in candidate_nodes:
        mu = (pos_acts[node].mean(dim=0) - neg_acts[node].mean(dim=0)).detach()
        mean_dirs[node] = mu
        delta_per_sample[node] = mu.unsqueeze(0).expand(n_neg, *mu.shape).contiguous()
    jac = adapter.jacobian_wrt_nodes(neg_train, candidate_nodes, target, target_sketch=None)
    cols: List[torch.Tensor] = []
    for node in candidate_nodes:
        delta = delta_per_sample[node]
        J = jac[node]
        if J.ndim == delta.ndim:
            J = J.unsqueeze(1)
        d_prime = int(J.shape[1])
        delta_exp = delta.unsqueeze(1).expand(-1, d_prime, *[-1] * (delta.ndim - 1))
        effect = (J * delta_exp).reshape(J.shape[0], d_prime, -1).sum(dim=-1)
        cols.append(effect.reshape(-1).float())
    G = torch.stack(cols, dim=1)
    return (G, q, mean_dirs)
