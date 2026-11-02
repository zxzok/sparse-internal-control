"""
Exact IOI and MMLU pair constructors. Role positions and original reference answer indices are retained.
"""
from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import torch
from torch import Tensor
_IOI_TEMPLATES = ['When {A} and {B} went to the {place}, {C} gave a {obj} to', 'After {A} and {B} went to the {place}, {C} gave a {obj} to', 'While {A} and {B} were at the {place}, {C} gave a {obj} to', 'When {A} and {B} went to the {place}, {C} decided to give a {obj} to']
_IOI_NAMES = ['John', 'Mary', 'Alice', 'Bob', 'Tom', 'Lisa', 'Mark', 'Kate', 'Paul', 'Jane', 'Mike', 'Sarah', 'David', 'Emma', 'James', 'Anna']
_IOI_PLACES = ['store', 'park', 'school', 'market', 'library', 'office']
_IOI_OBJECTS = ['ring', 'book', 'ball', 'drink', 'bottle', 'flower']

@dataclass
class IOIExample:
    clean_text: str
    corrupt_text: str
    io_name: str
    s_name: str

def _sample_ioi_example(rng: random.Random) -> IOIExample:
    template = rng.choice(_IOI_TEMPLATES)
    a, b = rng.sample(_IOI_NAMES, 2)
    place = rng.choice(_IOI_PLACES)
    obj = rng.choice(_IOI_OBJECTS)
    clean_text = template.format(A=a, B=b, place=place, C=a, obj=obj)
    corrupt_text = template.format(A=b, B=a, place=place, C=b, obj=obj)
    return IOIExample(clean_text=clean_text, corrupt_text=corrupt_text, io_name=b, s_name=a)

def _leading_space_token_id(tokenizer: Any, word: str) -> int:
    ids = tokenizer.encode(' ' + word, add_special_tokens=False)
    return int(ids[0])

def build_ioi_pairs(tokenizer: Any, n: int, device: str, seed: int=0, max_len: Optional[int]=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build paired IOI clean/corrupt batches with position labels.

    Returns ``(clean, corrupt)`` dicts, each containing:
      ``tokens``           [batch, seq]
      ``attention_mask``   [batch, seq]
      ``io_token_ids``     [batch]   (leading-space tokens of the IO name)
      ``s_token_ids``      [batch]   (leading-space tokens of the S name)
      ``answer_pos``       [batch]   (index of last token before answer)
      ``positions``        dict of {"END": LongTensor[batch]} and optionally
                           {"IO": ..., "S1": ..., "S2": ...} when the name
                           tokens can be unambiguously located in the prompt.
    """
    rng = random.Random(seed)
    examples = [_sample_ioi_example(rng) for _ in range(n)]
    clean_texts = [ex.clean_text for ex in examples]
    corrupt_texts = [ex.corrupt_text for ex in examples]
    clean_enc = tokenizer(clean_texts, return_tensors='pt', padding=True)
    corrupt_enc = tokenizer(corrupt_texts, return_tensors='pt', padding=True)
    seq_len = min(clean_enc['input_ids'].shape[1], corrupt_enc['input_ids'].shape[1])
    if max_len is not None:
        seq_len = min(seq_len, max_len)

    def _truncate(enc: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        return {k: v[:, :seq_len].to(device) for k, v in enc.items()}
    clean_tok = _truncate(clean_enc)
    corrupt_tok = _truncate(corrupt_enc)

    def _end_pos(enc: Mapping[str, Tensor]) -> Tensor:
        mask = enc['attention_mask']
        return (mask.sum(dim=-1) - 1).to(torch.long)
    clean_end = _end_pos(clean_tok)
    corrupt_end = _end_pos(corrupt_tok)
    io_ids = torch.tensor([_leading_space_token_id(tokenizer, ex.io_name) for ex in examples], dtype=torch.long, device=device)
    s_ids = torch.tensor([_leading_space_token_id(tokenizer, ex.s_name) for ex in examples], dtype=torch.long, device=device)

    def _find_positions(tok: Tensor, name_id: int, occurrence: int) -> int:
        matches = (tok == name_id).nonzero(as_tuple=False).flatten().tolist()
        if len(matches) >= occurrence:
            return matches[occurrence - 1]
        return -1
    positions_clean: Dict[str, Tensor] = {'END': clean_end}
    positions_corrupt: Dict[str, Tensor] = {'END': corrupt_end}
    s1_clean, s2_clean, io_clean_pos = ([], [], [])
    s1_corr, s2_corr, io_corr_pos = ([], [], [])
    for i, ex in enumerate(examples):
        sid = int(s_ids[i].item())
        iid = int(io_ids[i].item())
        s1_clean.append(_find_positions(clean_tok['input_ids'][i], sid, 1))
        s2_clean.append(_find_positions(clean_tok['input_ids'][i], sid, 2))
        io_clean_pos.append(_find_positions(clean_tok['input_ids'][i], iid, 1))
        s1_corr.append(_find_positions(corrupt_tok['input_ids'][i], iid, 1))
        s2_corr.append(_find_positions(corrupt_tok['input_ids'][i], iid, 2))
        io_corr_pos.append(_find_positions(corrupt_tok['input_ids'][i], sid, 1))
    positions_clean['S1'] = torch.tensor(s1_clean, dtype=torch.long, device=device)
    positions_clean['S2'] = torch.tensor(s2_clean, dtype=torch.long, device=device)
    positions_clean['IO'] = torch.tensor(io_clean_pos, dtype=torch.long, device=device)
    positions_corrupt['S1'] = torch.tensor(s1_corr, dtype=torch.long, device=device)
    positions_corrupt['S2'] = torch.tensor(s2_corr, dtype=torch.long, device=device)
    positions_corrupt['IO'] = torch.tensor(io_corr_pos, dtype=torch.long, device=device)
    clean = {'tokens': clean_tok['input_ids'], 'attention_mask': clean_tok['attention_mask'], 'io_token_ids': io_ids, 's_token_ids': s_ids, 'answer_pos': clean_end, 'positions': positions_clean}
    corrupt = {'tokens': corrupt_tok['input_ids'], 'attention_mask': corrupt_tok['attention_mask'], 'io_token_ids': io_ids, 's_token_ids': s_ids, 'answer_pos': corrupt_end, 'positions': positions_corrupt}
    return (clean, corrupt)
_GT_NOUNS = ('war', 'festival', 'tournament', 'expedition', 'experiment', 'construction', 'concert', 'negotiation', 'trial', 'debate')

@dataclass
class MCQAExample:
    question: str
    choices: Sequence[str]
    correct_index: int
    distractor_index: int
_MMLU_STEM_SUBJECTS = ('abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'machine_learning')
_MMLU_HUMANITIES_SUBJECTS = ('formal_logic', 'high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'international_law', 'jurisprudence', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'prehistory', 'professional_law', 'world_religions')

def load_mmlu_examples(category: str, n: int, seed: int=0, split: str='test') -> List['MCQAExample']:
    """Load `n` MCQAExamples from `cais/mmlu` filtered to a subject category.

    `category` ∈ {"stem", "humanities", "other", "all"}. Picks a random
    `distractor_index` ≠ `correct_index` per example (deterministic by seed).
    Requires `pip install datasets` and network on first call (cached after).
    """
    from datasets import load_dataset
    rng = random.Random(seed)
    if category == 'stem':
        subjects = _MMLU_STEM_SUBJECTS
    elif category == 'humanities':
        subjects = _MMLU_HUMANITIES_SUBJECTS
    elif category == 'all':
        subjects = None
    else:
        subjects = None
    ds = load_dataset('cais/mmlu', 'all', split=split, revision=os.environ['SIC_DATASET_REVISION'])
    rows = list(ds)
    if subjects is not None:
        rows = [r for r in rows if r['subject'] in subjects]
    elif category == 'other':
        excl = set(_MMLU_STEM_SUBJECTS) | set(_MMLU_HUMANITIES_SUBJECTS)
        rows = [r for r in rows if r['subject'] not in excl]
    rng.shuffle(rows)
    rows = rows[:n]
    examples: List[MCQAExample] = []
    for r in rows:
        choices = list(r['choices'])
        correct = int(r['answer'])
        wrong = [i for i in range(len(choices)) if i != correct]
        rng.shuffle(wrong)
        examples.append(MCQAExample(question=r['question'], choices=choices, correct_index=correct, distractor_index=wrong[0]))
    return examples

def build_mcqa_pairs(tokenizer: Any, examples: Sequence[MCQAExample], device: str, choice_letters: Sequence[str]=('A', 'B', 'C', 'D'), prompt_template: str='Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:') -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build paired MCQA batches.

    Clean prompt ends with the real options; corrupt swaps the correct option
    with the distractor option so that the "Answer:" logits over
    ``{A, B, C, D}`` point to a different letter.

    The target metric is the 4-dim logit vector over ``choice_letters`` at the
    last token.
    """
    clean_texts, corrupt_texts = ([], [])
    for ex in examples:
        base_kwargs = {letter: ex.choices[i] for i, letter in enumerate(choice_letters)}
        clean_texts.append(prompt_template.format(question=ex.question, **base_kwargs))
        swapped = list(ex.choices)
        i, j = (ex.correct_index, ex.distractor_index)
        swapped[i], swapped[j] = (swapped[j], swapped[i])
        corrupt_kwargs = {letter: swapped[k] for k, letter in enumerate(choice_letters)}
        corrupt_texts.append(prompt_template.format(question=ex.question, **corrupt_kwargs))
    clean_enc = tokenizer(clean_texts, return_tensors='pt', padding=True)
    corrupt_enc = tokenizer(corrupt_texts, return_tensors='pt', padding=True)
    seq_len = min(clean_enc['input_ids'].shape[1], corrupt_enc['input_ids'].shape[1])

    def _last_nonpad(enc: Mapping[str, Tensor]) -> Tensor:
        mask = enc['attention_mask'][:, :seq_len]
        return (mask.sum(dim=-1) - 1).to(torch.long)
    choice_token_ids = torch.tensor([[_leading_space_token_id(tokenizer, letter) for letter in choice_letters] for _ in examples], dtype=torch.long, device=device)

    def _pack(enc: Mapping[str, Tensor]) -> Dict[str, Any]:
        last = _last_nonpad(enc).to(device)
        return {'tokens': enc['input_ids'][:, :seq_len].to(device), 'attention_mask': enc['attention_mask'][:, :seq_len].to(device), 'choice_token_ids': choice_token_ids, 'answer_pos': last, 'positions': {'ANSWER': last}}
    return (_pack(clean_enc), _pack(corrupt_enc))

def build_refusal_pairs(tokenizer: Any, harmful_prompts: Sequence[str], harmless_prompts: Sequence[str], device: str, refusal_words: Sequence[str]=('I', "I'm", 'I cannot', 'Sorry'), compliance_words: Sequence[str]=('Sure', 'Of', 'Here', 'Let'), chat_template: Optional[str]=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build paired harmful (clean for refusal) / harmless (corrupt) batches.

    ``clean_text_i = chat_template(harmful_prompts[i])``
    ``corrupt_text_i = chat_template(harmless_prompts[i])``
    Target metric = ``log p(refusal token | prompt) - log p(compliance token | prompt)``
    at the assistant-turn first position.
    """
    assert len(harmful_prompts) == len(harmless_prompts)

    def _format(p: str) -> str:
        if chat_template is None:
            return p
        return chat_template.format(prompt=p)
    clean_texts = [_format(p) for p in harmful_prompts]
    corrupt_texts = [_format(p) for p in harmless_prompts]
    clean_enc = tokenizer(clean_texts, return_tensors='pt', padding=True)
    corrupt_enc = tokenizer(corrupt_texts, return_tensors='pt', padding=True)
    seq_len = min(clean_enc['input_ids'].shape[1], corrupt_enc['input_ids'].shape[1])

    def _last(enc: Mapping[str, Tensor]) -> Tensor:
        mask = enc['attention_mask'][:, :seq_len]
        return (mask.sum(dim=-1) - 1).to(torch.long)

    def _tok_ids(words: Sequence[str]) -> Tensor:
        ids = [_leading_space_token_id(tokenizer, w) for w in words]
        return torch.tensor(ids, dtype=torch.long, device=device)
    refusal_ids = _tok_ids(refusal_words)
    compliance_ids = _tok_ids(compliance_words)

    def _pack(enc: Mapping[str, Tensor]) -> Dict[str, Any]:
        last = _last(enc).to(device)
        b = enc['input_ids'].shape[0]
        return {'tokens': enc['input_ids'][:, :seq_len].to(device), 'attention_mask': enc['attention_mask'][:, :seq_len].to(device), 'refusal_token_ids': refusal_ids.expand(b, -1), 'compliance_token_ids': compliance_ids.expand(b, -1), 'answer_pos': last, 'positions': {'ANSWER': last}}
    return (_pack(clean_enc), _pack(corrupt_enc))
