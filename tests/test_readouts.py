import pytest
torch=pytest.importorskip('torch')
from sparse_control.tasks.mcqa import correct_minus_mean_wrong
from sparse_control.tasks.ioi import ioi_margin
from sparse_control.selection.core import flatten_target

def test_fixed_answer_reference():
 x=torch.tensor([[1.,2.,3.,7.],[7.,2.,3.,1.]])
 assert torch.allclose(correct_minus_mean_wrong(x,[3,3]),torch.tensor([5.,-3.]))
 assert flatten_target(x).tolist()==[1,2,3,7,7,2,3,1]
def test_ioi_reference_tokens():
 x=torch.tensor([[0.,7.,2.],[0.,1.,4.]])
 assert ioi_margin(x,torch.tensor([1,1]),torch.tensor([2,2])).tolist()==[5,-3]
def test_wordsets_are_preserved():
 from sparse_control.tasks.refusal import _REFUSAL_WORDS,_COMPLIANCE_WORDS
 from sparse_control.tasks.sentiment import _POS_WORDS,_NEG_WORDS
 assert _REFUSAL_WORDS==['I','Sorry','Unfortunately','Cannot','Can']
 assert _POS_WORDS==['good','great','wonderful','amazing','happy','excellent','fantastic','lovely'] and _NEG_WORDS==['bad','terrible','awful','horrible','miserable','disappointing','boring','depressing']
