import pytest
torch=pytest.importorskip('torch')
from sparse_control.adapters.positions import resolve_position,gather_at_positions,scatter_add_at_positions
from sparse_control.adapters.transformerlens import _resolve_position
from sparse_control.selection.core import DriverNode
def test_explicit_padding_rules():
 m=[[1,1,0,0],[0,0,1,1],[1,1,1,1]]
 assert resolve_position(m,'last_valid').tolist()==[1,3,3]
 assert resolve_position(m,'sum_mask_minus_one').tolist()==[1,1,3]
 assert resolve_position(m,'last_tensor').tolist()==[3,3,3]
def test_separate_member_positions():
 n=DriverNode('head_pos',0,'ANSWER',0);a={'tokens':torch.ones((1,4)), 'positions':{'ANSWER':[1]}};b={'tokens':torch.ones((1,4)),'positions':{'ANSWER':[3]}}
 assert _resolve_position(n,a).item()==1 and _resolve_position(n,b).item()==3
 x=torch.zeros((2,4,3));p=torch.tensor([1,3]);d=torch.ones((2,3));y=scatter_add_at_positions(x,p,d)
 assert torch.equal(gather_at_positions(y,p),d) and x.sum()==0 and y.sum()==6
