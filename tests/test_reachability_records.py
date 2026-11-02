from collections import defaultdict
from sparse_control.io import load_jsonl
from sparse_control.metrics.reachability import rel_tar

def test_all_formal_vectors(root):
 r=load_jsonl(root/'data/measurements/formal_reachability/readouts.jsonl');g=defaultdict(list)
 for x in r:g[x['series'],x['method'],x['direction']].append(rel_tar(x['source_behavior'],x['target_behavior'],x['forward_behavior']))
 assert len(r)==272 and len(g)==39 and len(g)*5==195
 ioi=next(v for k,v in g.items() if k[0].startswith('ioi') and k[1]=='driver_omp');assert sum(x<=.4 for x in ioi)==10
 for k,v in g.items():
  if k[2]=='F':assert not any(x<=.5 for x in v)
