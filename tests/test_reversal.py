from sparse_control.metrics.reversal import reversal_metrics
import pytest
def test_target_side_ratio_and_degeneracies():
 r=reversal_metrics([0],[3],[1]);assert r['ratio']==2 and r['passes']
 assert reversal_metrics([0],[3],[0])['status']=='positive_infinity'
 assert reversal_metrics([0],[0],[0])['passes'] is None
 assert reversal_metrics([0],[3],[3])['ratio']==0
