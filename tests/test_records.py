import pytest
from sparse_control.records import *
def test_invalid_records():
 with pytest.raises(ValueError):unique_ids([{'record_id':'a'},{'record_id':'a'}])
 with pytest.raises(ValueError):InterventionRecord('a',['x'],[],'P').validate()
 with pytest.raises(ValueError):InputRecord('x',['p'],[[1,0]],[[1,0]],[1]).validate()
 with pytest.raises(ValueError):ReadoutRecord('r','i',['p'],[1,2],[1],[1,2],[1,2]).validate()
 with pytest.raises(ValueError):ReadoutRecord('r','i',['p'],[1],[1],[float('nan')],[2]).validate()
def test_empty_support_is_valid():assert InterventionRecord('zero',[],[],'F',0).validate()
