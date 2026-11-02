"""Check record coverage, release paths and distributed checksums."""
import ast,hashlib,json,re
from urllib.parse import unquote, urlsplit
from pathlib import Path
from collections import Counter
from ..io import load_json,load_jsonl,resolve_path
from ..registry import validate_registry,load_experiments
from ..records import InterventionRecord,unique_ids

def check_records(root):
 root=Path(root);ex=load_experiments(root);total=0
 for eid,e in ex.items():
  c=load_json(root/e['config_path'])
  if eid.startswith('sentiment') and (c.get('feedback')!='not_evaluated' or 'feedback' in c['available_stages']):raise ValueError('Sentiment has no registered feedback experiment')
  for rel in e['record_paths']:
   if rel.endswith('.jsonl'):
    rs=load_jsonl(root/rel);unique_ids(rs);total+=len(rs)
  if c.get('interventions'):
   rs=load_jsonl(root/c['interventions']);unique_ids(rs,'intervention_id')
   for r in rs:InterventionRecord(r['intervention_id'],r['candidates'],r['coefficients'],r['direction_mode'],r['nominal_dose']).validate()
  for rel in load_json(root/e['input_manifest'])['files']:resolve_path(root,rel).stat()
 formal=load_jsonl(root/'data/measurements/formal_reachability/readouts.jsonl')
 if len(formal)!=272:raise ValueError('Expected 272 complete nominal readout vectors')
 counts=Counter((r['series'],r['method'],r['direction']) for r in formal)
 if len(counts)!=39:raise ValueError('Expected 39 formal target-residual groups')
 for key,n in counts.items():
  gg=[r for r in formal if (r['series'],r['method'],r['direction'])==key];expected=10 if key[0].startswith('ioi') else 3 if 'module' in key[0] else 8
  if sorted(r['seed'] for r in gg)!=list(range(expected)):raise ValueError(f'Incomplete formal seed coverage: {key}')
  for r in gg:
   n=len(r['source_behavior'])
   if not n or any(len(r[k])!=n for k in ['target_behavior','forward_behavior']):raise ValueError('Incomplete readout vector')
   controls=load_jsonl(root/f"data/interventions/{r['source_experiment_id']}/interventions.jsonl");matches=[c for c in controls if c['intervention_id']==r['intervention_id']]
   if len(matches)!=1 or matches[0]['candidates']!=r['drivers'] or matches[0]['coefficients']!=r['coefficients']:raise ValueError(f'Intervention binding mismatch: {r["record_id"]}')
 for task in ['mcqa_humanities','mcqa_stem','refusal']:
  r=load_jsonl(root/f'data/measurements/{task}_head/dose_scan.jsonl');keys=[(x['seed'],x['algo'],x['mode'],x['strength']) for x in r]
  if len(keys)!=480 or len(set(keys))!=480:raise ValueError('Independent dose coverage')
  r=load_jsonl(root/f'data/measurements/{task}_head/feedback_trajectories.jsonl');expected=240 if task=='refusal' else 288
  if len(r)!=expected:raise ValueError('Feedback endpoint coverage')
 for task in ['mcqa_humanities','mcqa_stem','sentiment','refusal']:
  rs=load_jsonl(root/f'data/measurements/preservation/{task}.jsonl');keys=[(r['seed'],r.get('algo',r.get('method')),r['prompt_id']) for r in rs]
  if len(keys)!=768 or len(set(keys))!=768:raise ValueError('Preservation prompt/seed coverage')
 return dict(experiments=len(ex),jsonl_rows=total,formal_vectors=len(formal),formal_groups=len(counts))

def check_release(root,manifest=False):
 root=Path(root).resolve();reg=validate_registry(root);records=check_records(root)
 deny=re.compile(r'/[U]sers/|/[h]ome/(?:a00|xizhe)|(?:sk-proj-|hf_)[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----')
 errors=[];files=[]
 for p in root.rglob('*'):
  if p.is_dir():continue
  rel=p.relative_to(root).as_posix()
  if '.git' in Path(rel).parts:continue
  if '__pycache__' in p.parts or p.suffix=='.pyc' or '.egg-info' in rel or '.pytest_cache' in rel:continue
  if p.is_symlink():errors.append(f'Symlink: {rel}');continue
  if not p.stat().st_size:errors.append(f'Empty file: {rel}')
  if p.suffix in ['.py','.json','.jsonl','.md','.toml','.txt','.cff']:
   s=p.read_text(encoding='utf8')
   if deny.search(s):errors.append(f'Private path/credential pattern: {rel}')
   if p.suffix=='.py':ast.parse(s,filename=rel)
   if p.suffix=='.md':
    for link in re.findall(r'\]\(([^)]+)\)',s):
     target=link.strip('<>');url=urlsplit(target)
     if url.scheme or not url.path:continue
     dest=(p.parent/unquote(url.path)).resolve()
     if not dest.is_relative_to(root) or not dest.exists():errors.append(f'Broken documentation link: {rel}: {target}')
  files.append(rel)
 if manifest:
  allowed=load_allowlist(root);actual=set(files)
  if actual!=set(allowed):errors.append('Allowlist differs: '+str(sorted(actual^set(allowed))[:10]))
  entries={}
  for line in (root/'MANIFEST.sha256').read_text().splitlines():
   digest,rel=line.split('  ',1)
   if rel in entries:errors.append('Duplicate manifest entry: '+rel)
   entries[rel]=digest
   if hashlib.sha256(resolve_path(root,rel).read_bytes()).hexdigest()!=digest:errors.append('Checksum mismatch: '+rel)
  if set(entries)!=set(allowed)-{'MANIFEST.sha256'}:errors.append('Manifest does not cover exactly the release allowlist')
  for artifact in load_json(root/'experiments/artifacts.json'):
   for rel in artifact['output_files']:
    if not resolve_path(root,'results/'+rel).is_file():errors.append('Missing artifact output: '+rel)
 if errors:raise ValueError('; '.join(errors))
 return dict(status='PASS',**reg,records=records,files=len(files),manifest_checked=manifest)

def load_allowlist(root):
 paths=(Path(root)/'release_files.txt').read_text().splitlines()
 if len(paths)!=len(set(paths)) or not paths:raise ValueError('Invalid release allowlist')
 for rel in paths:
  p=resolve_path(root,rel)
  if not p.is_file() or p.is_symlink():raise ValueError(f'Invalid release member: {rel}')
 return paths
