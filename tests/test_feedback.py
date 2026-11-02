from sparse_control.metrics.feedback import *
def test_terminal_not_best():
 r=endpoint_metrics(1,1.5,1,.1,4)
 assert not r['hit'] and r['initial_hit']
 assert update_dose(0,2,1,.15)==.15
 assert summarize_feedback([r])['initial_hits']==1

def test_targets_share_recorded_fit(root):
 from sparse_control.io import load_jsonl
 r=load_jsonl(root/'data/measurements/refusal_head/feedback_trajectories.jsonl')
 for seed in range(8):
  for algo in ['driver_omp','coverage_driver','top_attribution']:
   gg=[x for x in r if x['seed']==seed and x['algo']==algo];assert len(gg)==10
