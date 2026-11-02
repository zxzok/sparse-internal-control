import pytest
from sparse_control.metrics.statistics import *
def test_sample_sd_missing_and_threshold_precision():
 assert seed_summary([1,2,3])['sample_sd']==1
 with pytest.raises(ValueError):seed_summary([1,2],[0,1],[0,1,2])
 with pytest.raises(ValueError):seed_summary([1,2],[0,0])
 assert threshold_counts([.40004,.39996],.4)==1
