"""Exercise actual pruning and the explicit unsupported preservation boundary."""
import importlib
from types import SimpleNamespace
import pytest
torch=pytest.importorskip('torch')


@pytest.mark.parametrize('module',['core','module_reuse'])
@pytest.mark.parametrize('keep_last',[True,False])
def test_pruned_result_indices_follow_retained_nodes(module,keep_last):
    m=importlib.import_module('sparse_control.selection.'+module)
    nodes=[m.DriverNode('head_pos',0,'ANSWER',i) for i in range(3)]
    adapter=SimpleNamespace(run_clean_or_corrupt=lambda *args:torch.ones(3))
    discovery=m.DriverCircuitDiscovery(adapter,None,nodes,epsilon=.4,max_drivers=3,ridge=0,lin_tol=0,min_marginal_gain=0,backward_prune=True,fallback_mode='none')
    discovery.estimate_local_effects=lambda *args:(torch.eye(3),torch.tensor([3.,2.,1.]),{})
    discovery.validate_driver_set=lambda batch,target,drivers,deltas,strengths:((nodes[-1] in drivers) if keep_last else True,{'relative_error':0.})
    result=discovery.run({}, {})
    expected=[2] if keep_last else []
    assert result.selected_indices==expected
    assert result.drivers==[nodes[i] for i in result.selected_indices]
    assert len(result.strengths)==len(result.drivers)
    assert result.strengths.tolist()==([1.] if keep_last else [])
    assert any(x['action']=='try_remove' and x['ok'] for x in result.validation_log)


def test_unsupported_off_target_rejected_before_discovery():
    from sparse_control.selection.module_reuse import build_local_tables,DriverNode
    calls=[]
    def estimate(*args):
        calls.append(args)
        return torch.tensor([[2.],[2.]]),torch.ones(2),{node:torch.ones(2,1)}
    node=DriverNode('head_pos',0,'ANSWER',0)
    discovery=SimpleNamespace(estimate_local_effects=estimate,candidate_nodes=[node])
    with pytest.raises(NotImplementedError,match='Off-target'):
        build_local_tables(discovery,{}, {},off_target_batch={})
    assert calls==[]
    ordinary=build_local_tables(discovery,{}, {})
    assert len(calls)==1 and ordinary['H'] is None
    assert torch.allclose(ordinary['q'],torch.ones(2)/2**.5)
    assert ordinary['W'].tolist()==[[1.]]
