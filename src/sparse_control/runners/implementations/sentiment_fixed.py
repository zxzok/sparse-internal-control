"""
Task-specific executed experiment implementation.
"""
from __future__ import annotations
from sparse_control.runners.model_loading import load_tl, load_hf, load_tokenizer
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gpt2')
    p.add_argument('--device', default='auto')
    p.add_argument('--ridge', type=float, default=1.0)
    p.add_argument('--ks', default='3,5,8')
    p.add_argument('--seeds', default='0', help='Comma-separated seeds (shuffles prompt order)')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    seeds = [int(s) for s in args.seeds.split(',')]
    from transformer_lens import HookedTransformer
    model = load_tl(args.model, device=args.device, revision=os.environ['SIC_MODEL_REVISION'])
    tokenizer = model.tokenizer
    pos_ids = torch.tensor([tokenizer.encode(' ' + w, add_special_tokens=False)[0] for w in _POS_WORDS], device=args.device)
    neg_ids = torch.tensor([tokenizer.encode(' ' + w, add_special_tokens=False)[0] for w in _NEG_WORDS], device=args.device)
    target = SentimentLogitDiff(pos_ids, neg_ids)
    candidate_nodes: List[DriverNode] = []
    for L in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            candidate_nodes.append(DriverNode(basis='head_pos', layer=L, position='ANSWER', component=h))
    adapter = TransformerLensAdapter(model, device=args.device)
    rows = []
    algos = ['driver_omp', 'coverage_driver', 'top_attribution', 'random_matched']
    import random as _rnd
    for seed in seeds:
        torch.manual_seed(seed)
        rng_p = _rnd.Random(seed)
        idx_perm = list(range(len(_POS_PROMPTS)))
        rng_p.shuffle(idx_perm)
        pos_p = [_POS_PROMPTS[i] for i in idx_perm]
        neg_p = [_NEG_PROMPTS[i] for i in idx_perm]
        pos_batch = _build_batch(tokenizer, pos_p, args.device)
        neg_batch = _build_batch(tokenizer, neg_p, args.device)
        n_pp = len(_POS_PROMPTS)
        half = n_pp // 2
        train_idx = torch.arange(half, device=args.device)
        test_idx = torch.arange(half, n_pp, device=args.device)

        def slice_b(b, idxs):
            return {'tokens': b['tokens'][idxs], 'attention_mask': b['attention_mask'][idxs], 'answer_pos': b['answer_pos'][idxs], 'positions': {'ANSWER': b['positions']['ANSWER'][idxs]}}
        pos_train = slice_b(pos_batch, train_idx)
        pos_test = slice_b(pos_batch, test_idx)
        neg_train = slice_b(neg_batch, train_idx)
        neg_test = slice_b(neg_batch, test_idx)
        disc = DriverCircuitDiscovery(adapter=adapter, target_metric=target, candidate_nodes=candidate_nodes, epsilon=0.1, max_drivers=12, ridge=args.ridge)
        G_tr, q_tr, deltas_tr = disc.estimate_local_effects(pos_train, neg_train)
        G_te, q_te, deltas_te = disc.estimate_local_effects(pos_test, neg_test)
        pos_test_t = adapter.run_clean_or_corrupt(pos_test, 'clean', target)
        neg_test_t = adapter.run_clean_or_corrupt(neg_test, 'corrupt', target)
        print(f'\n=== seed {seed}: pos_test={pos_test_t.mean():+.3f}, neg_test={neg_test_t.mean():+.3f} ===')
        for kk in [int(s) for s in args.ks.split(',')]:
            for algo in algos:
                n_cand = G_tr.shape[1]
                if algo == 'driver_omp':
                    sel, _, _ = orthogonal_matching_pursuit(G_tr, q_tr, max_drivers=kk, ridge=args.ridge)
                elif algo == 'coverage_driver':
                    sel, _, _ = coverage_driver(G_tr, q_tr, max_drivers=kk, coverage_target=0.95)
                elif algo == 'top_attribution':
                    sel = torch.topk((G_tr.T @ q_tr).abs(), min(kk, n_cand)).indices.tolist()
                elif algo == 'random_matched':
                    rng = torch.Generator(device='cpu').manual_seed(99 + seed)
                    sel = torch.randperm(n_cand, generator=rng).tolist()[:kk]
                if not sel:
                    continue
                drivers = [candidate_nodes[i] for i in sel]
                alphas = ridge_solve(G_tr[:, sel], q_tr, ridge=args.ridge)
                patches_pe = {n: float(alphas[j].item()) * deltas_te[n] for j, n in enumerate(drivers)}
                pe_targval = adapter.run_with_patches(neg_test, patches_pe, target).detach()
                test_size = test_idx.shape[0]
                mu = {n: _broadcast_mean(deltas_tr[n], test_size) for n in drivers}
                patches_mu = {n: float(alphas[j].item()) * mu[n] for j, n in enumerate(drivers)}
                steer_targval = adapter.run_with_patches(neg_test, patches_mu, target).detach()
                rows.append({'seed': seed, 'k': kk, 'algo': algo, 'n_drivers': len(drivers), 'drivers': [n.label() for n in drivers], 'alphas': alphas.detach().cpu().tolist(), 'pos_test_target': pos_test_t.mean().item(), 'neg_test_target': neg_test_t.mean().item(), 'per_example_restored': pe_targval.mean().item(), 'held_out_steered': steer_targval.mean().item(), 'pe_recovery': (pe_targval.mean().item() - neg_test_t.mean().item()) / max(pos_test_t.mean().item() - neg_test_t.mean().item(), 1e-06), 'steer_recovery': (steer_targval.mean().item() - neg_test_t.mean().item()) / max(pos_test_t.mean().item() - neg_test_t.mean().item(), 1e-06)})
                if kk == int(args.ks.split(',')[-1]):
                    print(f"  seed={seed} k={kk} {algo:>20s}  pe={rows[-1]['pe_recovery'] * 100:.0f}%  steer={rows[-1]['steer_recovery'] * 100:.0f}%")
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'\nWrote {args.out}')
if __name__ == '__main__':
    main()
