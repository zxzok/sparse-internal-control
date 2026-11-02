import pytest
torch=pytest.importorskip('torch')
def test_hook_restores_model_and_zero_dose():
 tl=pytest.importorskip('transformer_lens')
 from sparse_control.adapters.transformerlens import TransformerLensAdapter
 from sparse_control.selection.core import DriverNode
 torch.manual_seed(0);m=tl.HookedTransformer(tl.HookedTransformerConfig(n_layers=1,d_model=16,n_ctx=8,d_head=4,n_heads=4,d_vocab=23,act_fn='relu',device='cpu',seed=0));m.eval();a=TransformerLensAdapter(m,'cpu')
 b={'tokens':torch.tensor([[1,2,3]]),'positions':{'ANSWER':torch.tensor([2])}}
 metric=lambda z,b:z[:,-1,:];n=DriverNode('head_pos',0,'ANSWER',0);u=torch.ones((1,4))
 y=a.run_clean_or_corrupt(b,'clean',metric)
 assert torch.allclose(y,a.run_with_patches(b,{n:0*u},metric))
 z=a.run_with_patches(b,{n:.3*u},metric)
 assert not torch.allclose(z,y)
 assert torch.allclose(y,a.run_clean_or_corrupt(b,'clean',metric))
