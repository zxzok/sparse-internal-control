"""MCQA readouts keep the original correct-option reference."""
from .pairs import MCQAExample, load_mmlu_examples, build_mcqa_pairs
load_mmlu_panel=load_mmlu_examples
def correct_minus_mean_wrong(option_logits, correct_index):
    import torch
    if option_logits.shape[-1]!=4: raise ValueError("Expected four option logits")
    v=option_logits.gather(-1,torch.as_tensor(correct_index,device=option_logits.device).unsqueeze(-1)).squeeze(-1)
    return v-(option_logits.sum(-1)-v)/3
def option_logits(logits, option_ids):
    return logits[..., option_ids]
