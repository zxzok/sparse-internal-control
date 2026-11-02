"""Configured execution with immutable input records and explicit profiles."""
import importlib.metadata,json,os,platform,subprocess,sys,time
from pathlib import Path
from ..io import load_json,write_json,output_directory
def execute(root,experiment,stage,seeds,profile_id,output,interventions=None,smoke=False,plan=False,task=None):
    root=Path(root);config=load_json(root/'configs'/(experiment+'.json'));profiles=load_json(root/'experiments/execution_profiles.json')
    if profile_id not in profiles:raise ValueError(f'Unknown execution profile: {profile_id}')
    profile=profiles[profile_id]
    if profile_id not in config['execution_profiles']:raise ValueError('Profile incompatible with experiment loader/readout')
    seeds=config['seed_ids'] if seeds is None else list(map(int,seeds.split(',')))
    if not seeds or len(set(seeds))!=len(seeds) or not set(seeds)<=set(config['seed_ids']):raise ValueError('Select a nonempty registered seed subset')
    if not smoke and stage not in config['available_stages']:raise ValueError(f'{experiment}: stage {stage} is not registered; available={config["available_stages"]}')
    if interventions and (smoke or stage in ['full','fit','preservation','generation'] or experiment=='refusal_single_site'):
        raise ValueError('--interventions is supported only by fixed-control nominal/dose/reversal/feedback stages')
    if task is not None and stage!='preservation':
        raise ValueError('--task is only supported for preservation')
    resolved=dict(config=config,profile_id=profile_id,profile=profile,seeds=seeds,stage=stage,smoke=smoke,intervention_source='explicit_file' if interventions else config.get('interventions'))
    if plan:return resolved
    output=output_directory(output,root);write_json(output/'resolved_config.json',resolved)
    for lib,version in profile['versions'].items():
        actual=importlib.metadata.version(lib)
        if actual.split('+')[0]!=version.split('+')[0]:raise RuntimeError(f'Profile {profile_id} requires {lib}=={version}; installed {actual}')
    actual={x:importlib.metadata.version(x) for x in profile['versions']};start=time.time()
    if smoke and experiment not in ['ioi_nominal','mcqa_humanities_nominal','mcqa_stem_nominal']:raise ValueError('Smoke is limited to nominal head/module interfaces')
    if smoke:
        from .nominal import run_nominal
        report=run_nominal(config,profile,seeds,root,output,smoke=True)
    elif experiment=='refusal_single_site':
        from .single_site import run_single_site
        report=run_single_site(root,config,profile,seeds,stage,output)
    elif stage=='preservation':
        from .preservation import run_preservation
        report=run_preservation(root,config,profile,seeds,output,task)
    elif stage=='generation':
        from .refusal_generation import run_refusal_generation
        report=run_refusal_generation(root,config,profile,seeds,output)
    elif stage in ['nominal','dose','reversal','feedback']:
        from .fixed_intervention import evaluate_fixed_intervention
        report=evaluate_fixed_intervention(root,config,profile,seeds,stage,output,interventions)
    elif stage in ['full','fit']:
        recipe=config.get('full_run')
        if not recipe:raise ValueError('No complete fitting producer registered')
        assets=load_json(root/'experiments/model_assets.json');asset=assets['models'][config['model_asset_id']]
        env={**os.environ,'SIC_MODEL_REVISION':asset['revision'],'SIC_DATASET_REVISION':assets['datasets']['cais/mmlu']['revision'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONHASHSEED':'0','SIC_EXECUTION_PROFILE':json.dumps(profile)}
        argv=recipe['arguments'][:]
        def put(flag,value):
            nonlocal argv
            if flag in argv:idx=argv.index(flag);argv[idx+1]=str(value)
            else:argv.extend([flag,str(value)])
        put('--device',profile['backend']);commands=[]
        for seedset in ([[s] for s in seeds] if recipe['seed_argument']=='--seed' else [seeds]):
            recipe=config.get('fitting_recipes_by_seed',{}).get(str(seedset[0]),config['full_run']);argv=recipe['arguments'][:];put('--device',profile['backend'])
            put(recipe['seed_argument'],','.join(map(str,seedset)))
            name='seed'+str(seedset[0]) if len(seedset)==1 else 'series';dest=output/(name+'.jsonl')
            env['SIC_MODEL_METADATA']=str(output/(name+'_model_loads.jsonl'))
            if experiment=='ioi_backup':put('--ioi-jsonl',root/config['support_source'])
            command=[sys.executable,'-m',recipe['module'],*argv,recipe['output_flag'],str(dest)]
            with (output/(name+'.log')).open('w') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=root)
            commands.append({'module':recipe['module'],'seed_ids':seedset,'exit_code':result.returncode})
            if result.returncode:raise RuntimeError(f'{experiment}: producer failed; see {name}.log')
        observed=[]
        for path in sorted(output.glob('*_model_loads.jsonl')):
            observed.extend(json.loads(line) for line in path.read_text().splitlines())
        if not observed:raise RuntimeError('Producer did not record actual model-loading metadata')
        report=dict(model_execution=True,selection=True,coefficient_refit=True,commands=commands,stages='complete producer sequence',loaded_models=observed)
    else:raise ValueError(stage)
    report.update(experiment_id=experiment,stage=stage,profile_id=profile_id,versions=actual,python_version=platform.python_version(),elapsed_seconds=time.time()-start,status='COMPLETE')
    write_json(output/'run_metadata.json',report);return report
