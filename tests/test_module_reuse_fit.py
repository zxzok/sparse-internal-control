import pytest
torch=pytest.importorskip('torch')
from sparse_control.selection.module_reuse import ridge_solve

def test_sample_normalized_ridge():
 G=torch.tensor([[2.],[2.],[2.],[2.]]);q=torch.ones(4)
 b=ridge_solve(G/2,q/2,ridge=1)
 assert b.item()==pytest.approx(.4)
 assert abs(b.item()-ridge_solve(G,q,ridge=1).item())>.05

def test_reuse_conditions_use_distinct_coefficients(root):
 from sparse_control.io import load_jsonl
 r=load_jsonl(root/'data/interventions/mcqa_humanities_reuse/interventions.jsonl')
 assert len(r)==32 and {x['coefficient_condition'] for x in r}=={'refit','fully_frozen'}
