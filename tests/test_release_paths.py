import pytest
from sparse_control.io import resolve_path,load_jsonl,output_directory
from sparse_control.release.check import check_release

def test_path_and_input_protection(root,tmp_path):
 with pytest.raises(ValueError):resolve_path(root,'../outside')
 with pytest.raises(ValueError):output_directory(root/'data/new',root)
 p=tmp_path/'bad.jsonl';p.write_text('{bad}\n')
 with pytest.raises(ValueError):load_jsonl(p)
 assert check_release(root)['status']=='PASS'
