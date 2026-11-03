from sparse_control.cli import reproduce
from sparse_control.reporting.paper_checks import compare_with_expected

def test_independent_paper_values(root,tmp_path):
 reproduce(root,tmp_path)
 r=compare_with_expected(tmp_path,root/'results/paper_values.json');assert r['tables']==13 and r['numeric_rows']==162 and r['supporting_numbers']==15
