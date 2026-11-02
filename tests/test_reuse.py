import pytest
from sparse_control.metrics.reuse import *
def test_series_specific_denominators():
 assert mcqa_recovery([1],[0],[.5])==.5
 assert mcqa_recovery([1],[1+1e-10],[2]) is None
 assert sentiment_recovery([1],[0],[.5])==-500000
 d=ioi_recovery([1,10],[0,10]);assert d['recovery']==.5 and d['failure_fraction']==.5
