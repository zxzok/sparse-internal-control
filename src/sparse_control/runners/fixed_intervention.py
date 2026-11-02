"""Evaluate explicitly saved supports and coefficients without selection or refitting."""
import json,os,random,re,time
from pathlib import Path
from ..io import load_json,load_jsonl,write_json
from ..metrics.reachability import rel_tar,r_clean

def node(label):
    from ..selection.core import DriverNode
    m=re.fullmatch(r'([^:]+):L(\d+):P([^:]+)(?::H(\d+))?',label)
    if not m:raise ValueError(f'Invalid candidate label: {label}')
    return DriverNode(basis=m[1],layer=int(m[2]),position=m[3],component=int(m[4]) if m[4] else None)
def slice_batch(b,indices):
    from .implementations.mcqa_feedback import _slice_b
    return _slice_b(b,indices)
def build_panel(config,tokenizer,device,seed):
    import torch
    from ..tasks.pairs import build_ioi_pairs,build_mcqa_pairs,load_mmlu_examples
    from ..adapters.transformerlens import IOILogitDiff,MCQALogitVector
    exp=config['experiment_id'];torch.manual_seed(seed);random.seed(seed)
    if exp.startswith('ioi'):
        c,s=build_ioi_pairs(tokenizer,n=64,device=device,seed=seed);target=IOILogitDiff();return c,s,c,s,target
    if exp.startswith('mcqa') or exp.startswith('model_'):
        module=exp.endswith('_nominal') or exp.startswith('model_');n=8 if module else 24
        if exp.startswith('model_'):n=4
        examples=load_mmlu_examples(category='stem' if 'stem' in exp else 'humanities',n=n,seed=seed)
        c,s=build_mcqa_pairs(tokenizer,examples,device=device)
        if module:return c,s,c,s,MCQALogitVector()
        from .implementations.mcqa_feedback import MCQAMarginScalar
        c['correct_indices']=s['correct_indices']=torch.tensor([e.correct_index for e in examples],device=device)
        target=MCQAMarginScalar()
    elif exp.startswith('sentiment'):
        from ..tasks import sentiment as m
        order=list(range(len(m._POS_PROMPTS)));random.Random(seed).shuffle(order)
        c=m._build_batch(tokenizer,[m._POS_PROMPTS[i] for i in order],device);s=m._build_batch(tokenizer,[m._NEG_PROMPTS[i] for i in order],device)
        pos=[tokenizer.encode(' '+w,add_special_tokens=False)[0] for w in m._POS_WORDS];neg=[tokenizer.encode(' '+w,add_special_tokens=False)[0] for w in m._NEG_WORDS]
        target=m.SentimentLogitDiff(torch.tensor(pos,device=device),torch.tensor(neg,device=device));n=len(order)
    elif exp.startswith('refusal'):
        from ..tasks import refusal as m
        order=list(range(20));random.Random(seed).shuffle(order)
        c=m._build_batch(tokenizer,[m._HARMFUL_PROMPTS[i] for i in order],device);s=m._build_batch(tokenizer,[m._HARMLESS_PROMPTS[i] for i in order],device)
        pos=[tokenizer.encode(' '+w,add_special_tokens=False)[0] for w in m._REFUSAL_WORDS];neg=[tokenizer.encode(' '+w,add_special_tokens=False)[0] for w in m._COMPLIANCE_WORDS]
        target=m.RefusalLogitDiff(torch.tensor(pos,device=device),torch.tensor(neg,device=device));n=20
    else:raise ValueError(f'No fixed-intervention panel for {exp}')
    a=torch.arange(n//2,device=device);b=torch.arange(n//2,n,device=device)
    return slice_batch(c,a),slice_batch(s,a),slice_batch(c,b),slice_batch(s,b),target
def load_model(root,config,profile):
    import torch
    assets=load_json(Path(root)/'experiments/model_assets.json');mid=config['model_asset_id'];asset=assets['models'][mid]
    device=profile['backend'];dtype=getattr(torch,profile['dtype'])
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA profile requires CUDA hardware')
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS profile requires Apple MPS hardware')
    os.environ['SIC_MODEL_REVISION']=asset['revision'];os.environ['SIC_DATASET_REVISION']=assets['datasets']['cais/mmlu']['revision']
    from .model_loading import load_hf, load_tl, load_tokenizer
    from .input_checks import check_recorded_model
    identity = check_recorded_model(root, config, asset['revision'])
    if profile['loader']=='HuggingFace':
        from ..adapters.huggingface import HFHookAdapter
        tokenizer=load_tokenizer(mid, revision=asset['revision'], profile=profile)
        model=load_hf(mid, revision=asset['revision'], device=device, tokenizer=tokenizer, profile=profile)
        adapter=HFHookAdapter(model,tokenizer,device=device)
    else:
        from ..adapters.transformerlens import TransformerLensAdapter
        model=load_tl(mid,revision=asset['revision'],device=device,profile=profile)
        tokenizer=model.tokenizer;adapter=TransformerLensAdapter(model,device=device)
    model._sic_execution_metadata['saved_model_identity']=identity
    model.eval();return model,tokenizer,adapter
def native(x):return x.detach().cpu().double().reshape(-1).tolist()
def serial(x):
    import torch
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x
def contrast(adapter,clean,source,nodes,microbatch=1):
    import torch
    if microbatch==1:
        pieces={n:[] for n in nodes}
        for i in range(len(clean['tokens'])):
            idx=torch.tensor([i],device=clean['tokens'].device);c=adapter.cache_activations(slice_batch(clean,idx),nodes);s=adapter.cache_activations(slice_batch(source,idx),nodes)
            for n in nodes:pieces[n].append((c[n]-s[n]).detach())
        return {n:torch.cat(x,0) for n,x in pieces.items()}
    c=adapter.cache_activations(clean,nodes);s=adapter.cache_activations(source,nodes)
    return {n:(c[n]-s[n]).detach() for n in nodes}
def evaluate_fixed_intervention(root,config,profile,seeds,stage,output,interventions=None):
    import torch
    root,output=Path(root),Path(output);exp=config['experiment_id']
    if not config.get('fixed_evaluation_available'):raise ValueError(f'{exp}: fixed controls are not available')
    path=Path(interventions) if interventions else root/config['interventions']
    if not path.exists():raise FileNotFoundError('Intervention record required; evaluation never fits missing supports')
    controls=load_jsonl(path);refs=load_jsonl(root/f'data/measurements/{exp}/readouts.jsonl') if exp!='refusal_single_site' else [r for r in controls if r['intervention_id'] in config['fixed_evaluation_ids']]
    identities={r['intervention_id'] for r in (controls if interventions else refs)};controls=[r for r in controls if r['intervention_id'] in identities and r['seed'] in seeds]
    if not controls:raise ValueError('No matching saved intervention IDs')
    model,tokenizer,adapter=load_model(root,config,profile);outputs=[];batches=[];arrays={};checks=[];t0=time.time()
    from .input_checks import check_saved_inputs,check_baselines
    with torch.no_grad():
        for seed in seeds:
            ct,st,ce,se,target=build_panel(config,tokenizer,profile['backend'],seed)
            input_check=check_saved_inputs(root,exp,seed,(ct,st,ce,se))
            batch_id=f'{exp}.seed{seed}';batches.append(dict(input_id=batch_id,seed=seed,fit_target=serial(ct),fit_source=serial(st),evaluation_target=serial(ce),evaluation_source=serial(se),padding_side=tokenizer.padding_side,truncation_side=tokenizer.truncation_side,readout_metadata=serial(vars(target))))
            rs=[r for r in controls if r['seed']==seed];nodes=list(dict.fromkeys(node(n) for r in rs for n in r['candidates']))
            s=adapter.run_clean_or_corrupt(se,'corrupt',target);t=adapter.run_clean_or_corrupt(ce,'clean',target)
            baseline=check_baselines(native(s),native(t),refs,seed)
            checks.append(dict(seed=seed,inputs=input_check,baselines=baseline))
            # F reads fitting activations only; evaluation target activations are accessed only for P.
            fd=contrast(adapter,ct,st,nodes,config.get("direction_cache_microbatch",1)) if any(r['direction_mode']=='F' for r in rs) else None
            pd=contrast(adapter,ce,se,nodes,config.get("direction_cache_microbatch",1)) if any(r['direction_mode']=='P' for r in rs) else None
            for r in rs:
                ns=[node(n) for n in r['candidates']];mode=r['direction_mode'];d=pd if mode=='P' else {n:fd[n].mean(0,keepdim=True).expand(len(se['tokens']),*fd[n].shape[1:]) for n in ns}
                if mode=='F':
                    for j,n in enumerate(ns):arrays[f"seed{seed}_{r['method']}_{j}"]=fd[n].mean(0).cpu().numpy()
                patch={n:float(a)*d[n] for n,a in zip(ns,r['coefficients'])}
                base=dict(experiment_id=exp,seed=seed,method=r['method'],direction=mode,intervention_id=r['intervention_id'],input_id=batch_id,source_behavior=native(s),target_behavior=native(t),readout_shape=list(t.shape),support_refit=False,coefficients=r['coefficients'],drivers=r['candidates'])
                def forward(gamma,on_target=False):return adapter.run_with_patches(ce if on_target else se,{n:gamma*p for n,p in patch.items()},target)
                if stage=='nominal':
                    y=forward(1);outputs.append(dict(base,dose=1.,forward_behavior=native(y),Rel_tar=rel_tar(native(s),native(t),native(y)),R_clean=r_clean(native(t),native(y))))
                elif stage=='reversal':
                    from ..metrics.reversal import reversal_metrics
                    y=forward(-1,True);outputs.append(dict(base,dose=-1.,reversed_behavior=native(y),metrics=reversal_metrics(native(s),native(t),native(y))))
                elif stage=='dose':
                    for gamma in config.get('dose_grid',[-1.,-.5,0.,.25,.5,.75,1.,1.25,1.5,2.]):
                        y=forward(gamma);outputs.append(dict(base,dose=gamma,forward_behavior=native(y),mean_margin=float(y.mean())))
                elif stage=='feedback':
                    f=config['feedback']
                    for target_margin in f['targets']:
                        gamma=0.;history=[]
                        for step in range(f['max_updates']):
                            y=forward(gamma);margin=float(y.mean());error=margin-target_margin
                            history.append(dict(update=step,dose=gamma,margin=margin,error=error))
                            if abs(error)<=f['tolerance']:break
                            gamma-=f['gain']*error
                        y=forward(gamma);margin=float(y.mean());outputs.append(dict(base,target_margin=target_margin,dose=gamma,terminal_margin=margin,terminal_error=margin-target_margin,initial_hit=abs(history[0]['error'])<=f['tolerance'],evaluations=len(history)+1,history=history))
                else:raise ValueError(stage)
    for name,rows in [('measurements',outputs),('batches',batches)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    if arrays:
        import numpy as np
        np.savez_compressed(output/'training_mean_directions.npz',**arrays)
    return dict(stage=stage,records=len(outputs),elapsed_seconds=time.time()-t0,selection=False,coefficient_refit=False,model_execution=True,input_checks=checks,loaded_model=model._sic_execution_metadata)
