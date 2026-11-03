"""Saved-support 16-token greedy Refusal continuations."""
import json,random
from pathlib import Path
from ..io import load_jsonl
from .fixed_intervention import load_model,build_panel,node,contrast
from .implementations.generation_kernels import _generate_single

def run_refusal_generation(root,config,profile,seeds,output):
 import torch
 root,output=Path(root),Path(output);refs=load_jsonl(root/'data/measurements/refusal_generation/generation.jsonl');model,tok,adapter=load_model(root,config,profile);rows=[]
 with torch.no_grad():
  for seed in seeds:
   ct,st,ce,se,metric=build_panel(config,tok,profile['backend'],seed);rs=[r for r in refs if r['seed']==seed];ns=list(dict.fromkeys(node(v) for r in rs for v in r['drivers']));ds=contrast(adapter,ct,st,ns,microbatch=len(ct['tokens']))
   for r in rs:
    drivers=[node(v) for v in r['drivers']];alpha=torch.tensor(r['alphas'],device=profile['backend']);sample=_generate_single(adapter,tok,r['prompt_text'],drivers,alpha,ds,metric);rows.append(dict(seed=seed,method=r['method'],prompt_id=r['prompt_id'],prompt_text=r['prompt_text'],drivers=r['drivers'],alphas=r['alphas'],direction='F',refit=False,**sample))
 (output/'generation.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));return dict(records=len(rows),model_execution=True,selection=False,coefficient_refit=False,loaded_model=model._sic_execution_metadata)
