import pytest
torch=pytest.importorskip('torch')
from sparse_control.selection.core import orthogonal_matching_pursuit,coverage_score_matrix,coverage_driver,DriverNode
from sparse_control.selection.baselines import random_matched_nodes

def test_raw_omp_cost_and_selected_stagnation():
 G=torch.eye(2);q=torch.tensor([1.,1.]);s,b,h=orthogonal_matching_pursuit(G,q,costs=torch.tensor([2.,1.]),max_drivers=1,ridge=1)
 assert s==[1] and b.item()==pytest.approx(.5)
 s,b,h=orthogonal_matching_pursuit(G,q,max_drivers=2,ridge=1000,min_marginal_gain=1)
 assert len(s)==1

def test_executed_coverage_floor_and_zero_rows():
 G=torch.tensor([[1e-6,0.],[0.,2.]]);q=torch.tensor([1e-6,2.]);A=coverage_score_matrix(G,q)
 assert A.tolist()==[[1.,0.],[0.,1.]]
 assert coverage_driver(G,q,max_drivers=1)[0]==[1]
 assert coverage_driver(G,torch.zeros(2),max_drivers=2)[0]==[]

def test_random_deterministic_budget():
 ns=[DriverNode('head_pos',0,'ANSWER',i) for i in range(12)]
 assert random_matched_nodes(ns,4,7)==random_matched_nodes(ns,4,7)
 assert len(set(random_matched_nodes(ns,4,7)))==4
