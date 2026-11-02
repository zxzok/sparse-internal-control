import math,pytest
from sparse_control.metrics.reachability import *
def test_distinct_denominators_and_vector_errors():
 s=[1,1];t=[2,2];p=[3,1]
 assert rel_tar(s,t,p)==pytest.approx(math.sqrt(2)/(math.sqrt(2)+1e-6))
 assert r_clean(t,p)==pytest.approx(.5)
 assert rel_tar(s,t,t)==0 and rel_tar(s,t,s)<1
 assert rel_tar(s,t,p)>.99
 with pytest.raises(ValueError):rel_tar([1],[1,2],[1,2])
