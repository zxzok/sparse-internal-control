"""Fixed 32-prompt preservation with task-specific direction access."""
import json
from pathlib import Path
from ..io import load_json,load_jsonl
from .fixed_intervention import load_model,build_panel,node,contrast,serial

def unpadded(tokenizer,prompt,device):
 import torch
 e=tokenizer(prompt,return_tensors='pt',padding=False,truncation=False);t=e['input_ids'].to(device);mask=e.get('attention_mask',torch.ones_like(t)).to(device);pos=torch.tensor([t.shape[1]-1],device=device)
 if t.shape[0]!=1 or not bool(mask.bool().all()):raise ValueError('Single unpadded prompt required')
 return dict(tokens=t,attention_mask=mask,answer_pos=pos,positions={'ANSWER':pos})
class FinalVocab:
 def __call__(self,logits,batch):
  import torch
  return logits[torch.arange(logits.shape[0],device=logits.device),batch['answer_pos'],:]

def run_preservation(root,config,profile,seeds,output,task=None,single_site=False):
 import numpy as np
 import torch
 from ..tasks.pairs import MCQAExample,build_mcqa_pairs
 from ..metrics.preservation import kl_unpatched_patched
 root,output=Path(root),Path(output);tasks=[task] if task else ['mcqa_humanities','mcqa_stem','sentiment','refusal'];panel=load_json(root/'data/inputs/preservation/panel.json')['records'];total=0
 for task in tasks:
  if task not in ['mcqa_humanities','mcqa_stem','sentiment','refusal']:raise ValueError(task)
  exp='sentiment_paired' if task=='sentiment' else task+'_head';cfg=load_json(root/f'configs/{exp}.json');model,tok,adapter=load_model(root,cfg,profile)
  records=load_jsonl(root/f'data/measurements/preservation/{task}.jsonl') if not single_site else load_jsonl(root/'data/measurements/refusal_single_site/preservation.jsonl')
  controls={}
  for r in records:
   if r['seed'] not in seeds:continue
   m=r.get('method',r.get('algo',r.get('method_label')));key=(r['seed'],m);controls[key]=r
  ns=list(dict.fromkeys(node(v) for r in controls.values() for v in r['drivers']));means={}
  dest=output/task;dest.mkdir();ld=dest/'logits';ld.mkdir();batches=[]
  with torch.no_grad(),(dest/'prompt_kl.jsonl').open('x') as f:
   if task in ['sentiment','refusal']:
    for seed in seeds:
     ct,st,ce,se,metric=build_panel(cfg,tok,profile['backend'],seed);ds=contrast(adapter,ct,st,ns,microbatch=len(ct['tokens']));means[seed]={n:x.mean(0,keepdim=True) for n,x in ds.items()};batches.append(dict(seed=seed,fit_target=serial(ct),fit_source=serial(st)))
   for index,p in enumerate(panel):
    batch=unpadded(tok,p['prompt'],profile['backend']);batches.append(dict(prompt_id=p['prompt_id'],batch=serial(batch)))
    baseline=adapter.run_clean_or_corrupt(batch,'unpatched',FinalVocab());np.savez_compressed(ld/f'baseline_{index:02d}.npz',logits=baseline.cpu().float().numpy())
    if task.startswith('mcqa'):
     ref=next(r for r in records if r['prompt_id']==p['prompt_id']);example=MCQAExample(p['question'],p['choices'],ref['off_companion_correct'],ref['off_companion_distractor']);c,s=build_mcqa_pairs(tok,[example],device=profile['backend'])
     if not torch.equal(c['tokens'],batch['tokens']):raise ValueError('Panel token identity mismatch')
     pd=contrast(adapter,c,s,ns,microbatch=1)
    for (seed,method),r in controls.items():
     ds=pd if task.startswith('mcqa') else means[seed];patch={node(v):float(a)*ds[node(v)] for v,a in zip(r['drivers'],r['alphas'])};y=adapter.run_with_patches(batch,patch,FinalVocab());v=kl_unpatched_patched(baseline.cpu().double().numpy().reshape(-1),y.cpu().double().numpy().reshape(-1));filename=f'patched_{index:02d}_{seed}_{method.replace(" ","_")}.npz';np.savez_compressed(ld/filename,logits=y.cpu().float().numpy());f.write(json.dumps(dict(task=task,seed=seed,method=method,prompt_id=p['prompt_id'],kl=float(v),baseline_logits=f'logits/baseline_{index:02d}.npz',patched_logits='logits/'+filename,drivers=r['drivers'],alphas=r['alphas'],dose=1.,direction='P' if task.startswith('mcqa') else 'F',refit=False))+'\n');total+=1
  (dest/'batches.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in batches));del model,adapter
 return dict(records=total,model_execution=True,selection=False,coefficient_refit=False)
