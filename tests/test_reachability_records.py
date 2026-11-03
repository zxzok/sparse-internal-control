from collections import defaultdict
from sparse_control.io import load_jsonl
from sparse_control.metrics.reachability import rel_tar
from sparse_control.io import load_json

def test_all_formal_vectors(root):
 r=load_jsonl(root/'data/measurements/formal_reachability/readouts.jsonl');g=defaultdict(list)
 for x in r:g[x['series'],x['method'],x['direction']].append(rel_tar(x['source_behavior'],x['target_behavior'],x['forward_behavior']))
 assert len(r)==272 and len(g)==39 and len(g)*5==195
 ioi=next(v for k,v in g.items() if k[0].startswith('ioi') and k[1]=='driver_omp');assert sum(x<=.4 for x in ioi)==10
 for k,v in g.items():
  if k[2]=='F':assert not any(x<=.5 for x in v)

def test_analysis_config_resolves_every_source_series(root):
 c=load_json(root/'configs/formal_reachability.json')
 records=load_jsonl(root/c['record_paths'][0])
 assert set(c['source_configs'])=={r['source_experiment_id'] for r in records}
 assert c['analysis_kind']=='saved_readout_vectors' and c['model_execution'] is False
 assert c['coefficient_refit'] is False and c['execution_profiles']==[]
 assert 'dtype' not in c and 'refit' not in c
 sources=[load_json(root/path) for path in c['source_configs'].values()]
 assert {s['experiment_id'] for s in sources}==set(c['source_configs'])
 assert {s['dtype'] for s in sources}=={'float32','bfloat16'}
 assert {s['refit']['lambda'] for s in sources}=={1.0,1e-5}
 expected=load_jsonl(root/'data/measurements/formal_reachability/readouts.jsonl')
 assert all(r['dose']==c['analysis']['nominal_dose'] for r in expected)
