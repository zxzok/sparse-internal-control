import pytest
from sparse_control.metrics.preservation import *
def test_kl_direction_full_vocab():
 np=pytest.importorskip('numpy',reason='Logit arithmetic requires the model environment; scalar replay does not')
 a=np.log([.9,.1]);b=np.log([.5,.5]);v=float(kl_unpatched_patched(a,b))
 assert v==pytest.approx(.9*np.log(1.8)+.1*np.log(.2))
 assert abs(v-float(kl_unpatched_patched(b,a)))>.1
 assert float(kl_unpatched_patched(a,a))==0

def test_nearest_rank_and_pass_aggregation():
 d=preservation_summary(list(range(1,257)),[0]*256,threshold=1000);assert d['p95_kl']==244
 d=preservation_summary([0,.0015],[0,0]);assert d['mean_kl_seed_passes']==1 and d['all_prompt_seed_passes']==0
