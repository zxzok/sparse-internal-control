"""Recompute paper tables from measurement records, never from expected values."""
import collections, math, statistics
from pathlib import Path
from ..io import load_json,load_jsonl,write_json,write_csv
from ..metrics.reachability import rel_tar,r_clean,distance,norm
from ..metrics.reversal import reversal_metrics
from ..metrics.dose import signed_spearman
from ..metrics.statistics import seed_summary
from ..metrics.feedback import endpoint_metrics,summarize_feedback
from ..metrics.preservation import preservation_summary

METHODS=['driver_omp','coverage_driver','top_attribution','random_matched','actual_gain_greedy']
LABELS=dict(zip(METHODS,['Driver-OMP','Coverage-Driver','Top-k','Random','Nonlinear greedy']))
TASKS={'ioi':'IOI','mcqa_humanities':'Humanities','mcqa_stem':'STEM','sentiment':'Sentiment','refusal':'Refusal'}
MODES={'per_example':'P','held_out_steer':'F','P':'P','F':'F'}
GRID=[-1.,-.5,0.,.25,.5,.75,1.,1.25,1.5,2.]
def records(root,exp,kind):return load_jsonl(Path(root)/'data/measurements'/exp/(kind+'.jsonl'))
def groups(rows,*keys):
    out=collections.defaultdict(list)
    for r in rows:out[tuple(r[k] for k in keys)].append(r)
    return out
def ms(values,digits=3):
    a=seed_summary(values)
    return f"{a['mean']:.{digits}f} ± {a['sample_sd']:.{digits}f}"
def count(n,d):return f'{n}/{d}'
def formal_results(root):
    raw=records(root,'formal_reachability','readouts');per=[]
    for r in raw:
        s,t,y=r['source_behavior'],r['target_behavior'],r['forward_behavior']
        row={k:r[k] for k in ['series','task','method','direction','seed']}
        row.update(Rel_tar=rel_tar(s,t,y),R_clean=r_clean(t,y),residual_norm=distance(t,y),displacement_norm=distance(t,s),target_norm=norm(t),unpatched_Rel_tar=rel_tar(s,t,s))
        row['passes_0p4']=row['Rel_tar']<=.4;per.append(row)
    summary=[];sensitivity=[]
    for (series,task,method,direction),rs in sorted(groups(per,'series','task','method','direction').items()):
        seeds=[r['seed'] for r in rs];n=10 if task=='ioi' else 3 if 'module' in series else 8
        x=seed_summary([r['Rel_tar'] for r in rs],seeds,list(range(n)))
        base=dict(series=series,task=task,method=method,direction=direction,seeds=n)
        summary.append(dict(base,Rel_tar_mean=x['mean'],Rel_tar_sample_sd=x['sample_sd'],passes_0p4=sum(r['passes_0p4'] for r in rs),R_clean_mean=statistics.fmean(r['R_clean'] for r in rs),R_clean_sample_sd=statistics.stdev(r['R_clean'] for r in rs)))
        for eps in [.3,.35,.4,.45,.5]:sensitivity.append(dict(base,epsilon=eps,passes=sum(r['Rel_tar']<=eps for r in rs)))
    if (len(per),len(summary),len(sensitivity))!=(272,39,195):raise ValueError('Formal reachability coverage mismatch')
    return per,summary,sensitivity
def feedback(root):
    rows=[]
    for task in ['mcqa_humanities','mcqa_stem','refusal']:
        tol=.1 if task=='refusal' else .3
        rs=records(root,task+'_head','feedback_trajectories')
        for (algo,mode),gg in groups(rs,'algo','mode').items():
            endpoints=[endpoint_metrics(r['M_star'],r['iter_final_M'],r['history'][0]['M'],tol,r['n_evals']) for r in gg]
            rows.append(dict(task=task,method=algo,direction=MODES[mode],**summarize_feedback(endpoints)))
    return rows
def preservation(root):
    rows=[]
    for task in ['mcqa_humanities','mcqa_stem','sentiment','refusal']:
        rs=records(root,'preservation',task)
        for algo in METHODS[:3]:
            gg=[r for r in rs if r.get('algo',r.get('method'))==algo]
            if len(gg)!=256 or len({(r['seed'],r['prompt_id']) for r in gg})!=256:raise ValueError('Preservation coverage')
            rows.append(dict(task=task,method=algo,**preservation_summary([r['kl'] for r in gg],[r['seed'] for r in gg])))
    return rows
def supporting_numbers(root):
    heat=load_json(Path(root)/'data/measurements/ioi_backup/pointwise_errors.json');a=heat['groups']['CORE']['rel_per_prompt'];b=heat['groups']['CORE+top6']['rel_per_prompt']
    recurrence=records(root,'ioi_recurrence','supports');freq=collections.Counter(n for r in recurrence for n in set(r['drivers']))
    gen=records(root,'refusal_generation','generation');proxy=records(root,'refusal_single_site','axes');kl=records(root,'refusal_single_site','preservation')
    result={'ioi_backup':dict(initial_mean=statistics.fmean(a),final_mean=statistics.fmean(b),first_addition_mean=statistics.fmean(heat['groups']['CORE+top1']['rel_per_prompt']),downward_crossings=sum(x>.3 and y<.3 for x,y in zip(a,b)),upward_crossings=sum(x<.3 and y>.3 for x,y in zip(a,b)),final_failures=sum(x>.3 for x in b),prompts=len(a),random_reference_size=heat['groups']['RANDOM']['size'] if 'RANDOM' in heat['groups'] else next(v['size'] for k,v in heat['groups'].items() if 'rand' in k.lower())),
       'ioi_recurrence':dict(configurations=len(recurrence),frequency_counts=dict(sorted(freq.items()))),
       'refusal_generation':dict(continuations=len(gen),exactly_16_tokens=sum(len(r['generated_token_ids'])==16 for r in gen),contains_end_think=sum('</think>' in r['generated_text_raw'] for r in gen),conditions=dict(collections.Counter(r['method'] for r in gen))),
       'refusal_prompt_passes':{r['method']:r['prompt_passes'] for r in preservation(root) if r['task']=='refusal'},
       'single_site':{}}
    for mode in ['per_example','held_out_steer']:
        rr=[r for r in proxy if r['axis']=='closed_loop_feedback' and r['mode']==mode]
        rv=[r for r in proxy if r['axis']=='formal_signed_reversal' and r['mode']==mode]
        result['single_site'][MODES[mode]]=dict(feedback_hits=sum(abs(r['iter_final_err'])<=.1 for r in rr),feedback_endpoints=len(rr),reversal_passes=sum(reversal_metrics(r['source_behavior'],r['target_behavior'],r['reversed_behavior'])['passes'] for r in rv),reversal_seeds=len(rv))
    result['single_site']['preservation']=dict(mean_kl=statistics.fmean(r['kl'] for r in kl),prompt_passes=sum(r['kl']<=.001 for r in kl),measurements=len(kl))
    return result

def build_table(artifact_id,root,output):
    root,output=Path(root),Path(output);cells=[];numeric=[];columns=[]
    if artifact_id=='nominal_reconstruction':
        columns=['Method','IOI (10 seeds)','Humanities (3 seeds)','STEM (3 seeds)']
        for m in METHODS:
            row=[LABELS[m]]
            for exp in ['ioi_nominal','mcqa_humanities_nominal','mcqa_stem_nominal']:
                rs=[r for r in records(root,exp,'nominal') if r['method']==m];v=[r['rel_l2_to_clean'] for r in rs]
                numeric.append(dict(experiment=exp,method=m,**seed_summary(v,[r['seed'] for r in rs],range(10 if exp=='ioi_nominal' else 3))))
                row.append(ms(v))
            cells.append(row)
    elif artifact_id in ('formal_reachability_samepanel','formal_reachability_heldout','reachability_tolerance'):
        per,summary,sensitivity=formal_results(root);index={(r['series'],r['method'],r['direction']):r for r in summary}
        def cell(series,m,d):
            r=index[series,m,d];return f"{r['Rel_tar_mean']:.3f} ± {r['Rel_tar_sample_sd']:.3f} ({r['passes_0p4']}/{r['seeds']})"
        if artifact_id=='formal_reachability_samepanel':
            series=['ioi_nominal_same_panel','mcqa_humanities_module_option_logits_same_panel','mcqa_stem_module_option_logits_same_panel'];columns=['Method','IOI','Humanities','STEM']
            cells=[[LABELS[m]]+[cell(s,m,'P') for s in series] for m in METHODS];numeric=[r for r in summary if r['series'] in series]
        elif artifact_id=='formal_reachability_heldout':
            columns=['Task','Method','P','F']
            for task,series in [('mcqa_humanities','mcqa_humanities_head_scalar_heldout'),('mcqa_stem','mcqa_stem_head_scalar_heldout'),('sentiment','sentiment_paired_heldout'),('refusal','refusal_head_scalar_heldout')]:
                for m in METHODS[:3]:cells.append([TASKS[task],LABELS[m],cell(series,m,'P'),cell(series,m,'F')])
            numeric=[r for r in summary if 'heldout' in r['series']]
        else:
            columns=list(sensitivity[0]);numeric=sensitivity;cells=[[r[k] for k in columns] for r in sensitivity]
        fd=output/'formal_reachability';fd.mkdir(exist_ok=True)
        for name,rows in [('per_seed',per),('summary',summary),('threshold_sensitivity',sensitivity)]:
            if not (fd/(name+'.csv')).exists():write_csv(fd/(name+'.csv'),rows)
    elif artifact_id in ('feedback_hits','feedback_details','feedback_tolerance'):
        fs=feedback(root);index={(r['task'],r['method'],r['direction']):r for r in fs}
        if artifact_id=='feedback_hits':
            columns=['Task','Mode']+[LABELS[m] for m in METHODS[:3]]
            rs=records(root,'ioi_feedback','feedback_trajectories');cells=[['IOI','P',count(sum(abs(r['iter_err'])<=.3 for r in rs),len(rs)),'---','---']]
            for task in ['mcqa_humanities','mcqa_stem','refusal']:
                for mode in ['P','F']:cells.append([TASKS[task],mode]+[count(index[task,m,mode]['hits'],index[task,m,mode]['total']) for m in METHODS[:3]])
            numeric=fs
        elif artifact_id=='feedback_details':
            columns=['Task','Method','Mode','Hits','MAE','Evals','Initial'];numeric=fs
            for r in fs:cells.append([TASKS[r['task']],LABELS[r['method']],r['direction'],count(r['hits'],r['total']),f"{r['mean_absolute_error']:.4f}",f"{r['mean_evaluations']:.3f}",count(r['initial_hits'],r['total'])])
        else:
            columns=['Task','Method','Mode','.10','.20','.30','.40','.50']
            for task in ['mcqa_humanities','mcqa_stem','refusal']:
                rs=records(root,task+'_head','feedback_trajectories')
                for m in METHODS[:3]:
                    for mode in ['per_example','held_out_steer']:
                        gg=[r for r in rs if r['algo']==m and r['mode']==mode];v=[abs(r['iter_final_err']) for r in gg]
                        cells.append([TASKS[task],LABELS[m],MODES[mode]]+[count(sum(x<=eps for x in v),len(v)) for eps in [.1,.2,.3,.4,.5]])
                        for eps in [.1,.2,.3,.4,.5]:numeric.append(dict(task=task,method=m,direction=MODES[mode],epsilon=eps,hits=sum(x<=eps for x in v),total=len(v)))
    elif artifact_id=='signed_reversal':
        columns=['Task','Mode','Method','Pass/seeds','Ratio mean ± SD']
        for exp,task in [('ioi_nominal','ioi'),('sentiment_paired','sentiment'),('mcqa_humanities_head','mcqa_humanities'),('mcqa_stem_head','mcqa_stem'),('refusal_head','refusal')]:
            for (m,mode),gg in groups(records(root,exp,'reversal'),'algo','mode').items():
                v=[reversal_metrics(r['source_behavior'],r['target_behavior'],r['reversed_behavior'])['ratio'] for r in gg]
                if any(x is None for x in v):raise ValueError('An infinite/undefined ratio requires explicit display')
                numeric.append(dict(task=task,method=m,direction=MODES[mode],passes=sum(x>=2 for x in v),**seed_summary(v)))
                cells.append([TASKS[task],MODES[mode],LABELS[m],count(sum(x>=2 for x in v),len(v)),ms(v,4)])
    elif artifact_id=='dose_ordering':
        columns=['Task','Mode','Method','rho>=.95','rho<=-.95','Mean','SD']
        for task in ['mcqa_humanities','mcqa_stem','refusal']:
            rs=records(root,task+'_head','dose_scan')
            for (m,mode),gg in groups(rs,'algo','mode').items():
                values=[]
                for (seed,),ss in groups(gg,'seed').items():
                    ss=sorted(ss,key=lambda r:r['strength'])
                    if [r['strength'] for r in ss]!=GRID:raise ValueError('Incomplete dose grid')
                    values.append(signed_spearman(GRID,[r['normalized_target_progress'] for r in ss]))
                d=seed_summary(values);pos=sum(v>=.95 for v in values);neg=sum(v<=-.95 for v in values)
                numeric.append(dict(task=task,method=m,direction=MODES[mode],positive=pos,negative=neg,**d))
                cells.append([TASKS[task],MODES[mode],LABELS[m],count(pos,8),count(neg,8),f"{d['mean']:.4f}",f"{d['sample_sd']:.4f}"])
    elif artifact_id=='off_target_preservation':
        columns=['Task','Method','Mean','Median','P95','Max','Mean seeds','All seeds'];numeric=preservation(root)
        for r in numeric:cells.append([TASKS[r['task']],LABELS[r['method']],f"{r['mean_kl']:.6f}",f"{r['median_kl']:.6f}",f"{r['p95_kl']:.6f}",f"{r['max_kl']:.4f}",count(r['mean_kl_seed_passes'],8),count(r['all_prompt_seed_passes'],8)])
    elif artifact_id=='mcqa_reuse':
        columns=['Task','Method','Seeds','Refit coefficients','Fully frozen']
        for task in ['mcqa_humanities','mcqa_stem']:
            rs=records(root,task+'_reuse','reuse')
            for m in METHODS[:4]:
                gg=[r for r in rs if r['algo']==m]
                cells.append([TASKS[task],LABELS[m],len(gg),ms(r['pe_recovery'] for r in gg),ms(r['steer_recovery'] for r in gg)])
                for mode,key in [('refit_P','pe_recovery'),('frozen_F','steer_recovery')]:numeric.append(dict(task=task,method=m,condition=mode,**seed_summary([r[key] for r in gg])))
    elif artifact_id=='ioi_reuse':
        columns=['Method','Refit recovery','Fixed-coefficient recovery','Refit failure fraction','Fixed failure fraction'];rs=records(root,'ioi_reuse','reuse')
        for m in METHODS[:4]:
            gg=[r for r in rs if r['algo']==m]
            cells.append([LABELS[m],ms(1-r['rel_l2_refit_mean'] for r in gg),ms(1-r['rel_l2_frozen_mean'] for r in gg),f"{statistics.fmean(r['frac_rel_gt_0p3_refit'] for r in gg):.3f}",f"{statistics.fmean(r['frac_rel_gt_0p3_frozen'] for r in gg):.3f}"])
            for mode in ['refit','frozen']:numeric.append(dict(method=m,condition=mode,**seed_summary([1-r['rel_l2_'+mode+'_mean'] for r in gg]),failure_fraction=statistics.fmean(r['frac_rel_gt_0p3_'+mode] for r in gg)))
    elif artifact_id=='sentiment_fixed_dose':
        columns=['Method','k','Mode','Mean','SD','Seeds'];rs=records(root,'sentiment_paired','fixed_dose')
        for m in METHODS[:4]:
            for k in [3,5,8]:
                gg=[r for r in rs if r['algo']==m and r['k']==k]
                for mode,key in [('P','pe_recovery'),('F','steer_recovery')]:
                    d=seed_summary([r[key] for r in gg]);numeric.append(dict(method=m,budget=k,direction=mode,**d));cells.append([LABELS[m],k,mode,f"{d['mean']:.4f}",f"{d['sample_sd']:.4f}",d['n']])
    elif artifact_id=='model_comparison':
        columns=['Model','Seeds']+[LABELS[m] for m in METHODS[:4]]
        for exp,label in [('mcqa_humanities_nominal','Qwen3-4B'),('model_qwen3_8b','Qwen3-8B'),('model_mistral7b','Mistral-7B')]:
            rs=records(root,exp,'nominal');row=[label,len({r['seed'] for r in rs})]
            for m in METHODS[:4]:
                gg=[r for r in rs if r['method']==m];v=[r['rel_l2_to_clean'] for r in gg];row.append(ms(v));numeric.append(dict(model=label,method=m,**seed_summary(v)))
            cells.append(row)
    elif artifact_id in ['control_axes','formal_reference_settings','experiment_matrix']:
        spec=load_json(root/'configs'/('display_'+artifact_id+'.json'));columns=spec['columns'];cells=spec['cells']
    elif artifact_id=='supporting_numbers':
        value=supporting_numbers(root);write_json(output/'supporting_numbers.json',value);return {'artifact_id':artifact_id,'scalars':value}
    else:raise ValueError(f'Unknown table {artifact_id}')
    result=dict(artifact_id=artifact_id,columns=columns,cells=cells,numeric_rows=numeric)
    td=output/'tables';td.mkdir(exist_ok=True);write_json(td/(artifact_id+'.json'),result)
    write_csv(td/(artifact_id+'.csv'),[dict(zip(columns,r)) for r in cells])
    if numeric:write_csv(td/(artifact_id+'_statistics.csv'),numeric)
    def tex(x):return str(x).replace('±',r'$\pm$').replace('_',r'\_').replace('>=',r'$\geq$').replace('<=',r'$\leq$')
    text='\\begin{tabular}{'+'l'*len(columns)+'}\n\\toprule\n'+' & '.join(map(tex,columns))+r' \\'+'\n\\midrule\n'+'\n'.join(' & '.join(map(tex,r))+r' \\' for r in cells)+'\n\\bottomrule\n\\end{tabular}\n'
    if artifact_id in ['control_axes','formal_reference_settings','experiment_matrix']:text=spec['latex_table']
    (td/(artifact_id+'.tex')).write_text(text)
    return result
