from sparse_control.release.check import check_records
from sparse_control.io import load_jsonl

def test_registered_coverage(root):assert check_records(root)['formal_vectors']==272

def test_other_series(root):
 expected={'ioi_nominal/nominal':50,'mcqa_humanities_nominal/nominal':15,'mcqa_stem_nominal/nominal':15,'sentiment_paired/fixed_dose':96,'ioi_recurrence/supports':25,'ioi_feedback/feedback_trajectories':24,'sentiment_unpaired_dose/dose_scan':84,'refusal_generation/generation':320,'refusal_single_site/preservation':256}
 for f,n in expected.items():assert len(load_jsonl(root/f'data/measurements/{f}.jsonl'))==n

def test_direction_stage_and_single_site_identity(root):
 from sparse_control.io import load_json
 for eid in ['sentiment_paired','sentiment_unpaired_dose']:
  c=load_json(root/f'configs/{eid}.json')
  assert c['feedback']=='not_evaluated' and 'feedback' not in c['available_stages']
 c=load_json(root/'configs/refusal_single_site.json')
 selected=set(c['fixed_evaluation_ids'])
 rows=[x for x in load_jsonl(root/c['interventions']) if x['intervention_id'] in selected]
 assert len(rows)==16 and len({(x['seed'],x['direction_mode']) for x in rows})==16
 assert all(len(x['candidates'])==1 for x in rows)
