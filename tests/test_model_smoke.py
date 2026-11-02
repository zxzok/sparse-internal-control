import json,os
from pathlib import Path
import pytest
@pytest.mark.smoke
def test_completed_bounded_model_runs():
 p=os.environ.get('SIC_SMOKE_OUTPUT_ROOT')
 if not p:pytest.skip('Set SIC_SMOKE_OUTPUT_ROOT to actual run outputs; replay does not execute pretrained models')
 for name in ['smoke_head','smoke_module']:
  d=Path(p)/name;r=json.loads((d/'run_metadata.json').read_text());m=json.loads((d/'measurements.jsonl').read_text().splitlines()[0])
  assert r['status']=='COMPLETE' and r['model_execution'] and r['smoke']
  assert m['seed']==0 and m['candidate_count']==8 and m['k']==1 and len(m['fit_source']['tokens'])==2
  assert len(m['source_behavior'])==len(m['target_behavior'])==len(m['forward_behavior'])
