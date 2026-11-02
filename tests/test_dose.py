import pytest
from sparse_control.metrics.dose import *
def test_signed_rank_ties_and_constant():
 assert signed_spearman([0,1,2],[3,2,1])==-1
 assert signed_spearman([0,1,2],[1,1,1]) is None
 assert ranks([1,1,2])==[1.5,1.5,3]
 assert signed_spearman([0,1,2],[0,0,1])==pytest.approx(.8660254037844386)
def test_mean_and_projection_are_distinct():
 assert vector_progress([0,0],[1,-1],[1,-1])>.999
 with pytest.raises(ValueError):mean_progress(0,0,1)
