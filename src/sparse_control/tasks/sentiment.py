"""
Paired Sentiment templates and first-subtoken word-set readout.
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Tuple
import torch
from sparse_control.selection.core import DriverCircuitDiscovery, DriverNode, ridge_solve, orthogonal_matching_pursuit, coverage_driver
from sparse_control.adapters.transformerlens import TransformerLensAdapter
_POS_PROMPTS = ['I just watched this movie and it was absolutely', 'The food at this restaurant is incredibly', 'She gave the best performance ever, truly', 'What a fantastic experience, I felt so', 'The team played brilliantly and ended up', 'After reading the book I was deeply', 'Their customer service is consistently', 'The whole event went smoothly and was', 'He won the award and was understandably', 'The new album is so well-produced and', 'She nailed the interview and emerged', 'Every dish at the dinner was simply', 'The view from the hotel balcony was', 'Her speech moved the audience and felt', 'The product worked exactly as promised, truly', 'The hotel staff were warm and attentive,']
_NEG_PROMPTS = ['I just watched this movie and it was utterly', 'The food at this restaurant is incredibly', 'She gave the worst performance ever, truly', 'What a dreadful experience, I felt so', 'The team played terribly and ended up', 'After reading the book I was deeply', 'Their customer service is consistently', 'The whole event went poorly and was', 'He lost the case and was understandably', 'The new album is so badly-produced and', 'She bombed the interview and emerged', 'Every dish at the dinner was simply', 'The view from the hotel balcony was', 'Her speech bored the audience and felt', 'The product failed exactly as feared, truly', 'The hotel staff were rude and dismissive,']
_POS_WORDS = ['good', 'great', 'wonderful', 'amazing', 'happy', 'excellent', 'fantastic', 'lovely']
_NEG_WORDS = ['bad', 'terrible', 'awful', 'horrible', 'miserable', 'disappointing', 'boring', 'depressing']

class SentimentLogitDiff:
    """Target = mean log P(POS_WORDS) - mean log P(NEG_WORDS) at last token."""

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
        pos_lp = log_p[:, pos].mean(dim=-1)
        neg_lp = log_p[:, neg].mean(dim=-1)
        return pos_lp - neg_lp

def _build_batch(tokenizer, prompts, device):
    enc = tokenizer(prompts, return_tensors='pt', padding=True)
    seq_len = enc['input_ids'].shape[1]
    last = (enc['attention_mask'].sum(dim=-1) - 1).to(torch.long).to(device)
    return {'tokens': enc['input_ids'].to(device), 'attention_mask': enc['attention_mask'].to(device), 'answer_pos': last, 'positions': {'ANSWER': last}}

def _broadcast_mean(delta: torch.Tensor, batch_size: int) -> torch.Tensor:
    return delta.mean(dim=0).unsqueeze(0).expand(batch_size, *delta.mean(dim=0).shape)


def build_paired_panel(tokenizer, device, seed):
    import random
    order=list(range(len(_POS_PROMPTS)));random.Random(seed).shuffle(order)
    return _build_batch(tokenizer,[_POS_PROMPTS[i] for i in order],device),_build_batch(tokenizer,[_NEG_PROMPTS[i] for i in order],device),order

def sentiment_margin(logits, batch, positive_ids, negative_ids):
    return SentimentLogitDiff(positive_ids,negative_ids)(logits,batch)
