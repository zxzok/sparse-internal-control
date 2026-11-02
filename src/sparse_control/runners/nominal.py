"""Fit and measure sparse interventions with the executed discovery kernels."""
def run_nominal(config,profile,seeds,root,output,smoke=False):
    import json,time,torch
    from pathlib import Path
    from .fixed_intervention import load_model,build_panel,native,serial
    from ..selection.core import DriverNode,DriverCircuitDiscovery
    from ..selection.baselines import coverage_driver_baseline,top_node_attribution_baseline,random_matched_baseline,actual_gain_greedy_baseline
    from ..metrics.reachability import rel_tar,r_clean
    model,tok,adapter=load_model(root,config,profile);rows=[]
    for seed in seeds:
        ct,st,ce,se,target=build_panel(config,tok,profile['backend'],seed)
        if smoke:
            from .fixed_intervention import slice_batch
            ix=torch.arange(min(2,len(ct['tokens'])),device=profile['backend']);ct,st=slice_batch(ct,ix),slice_batch(st,ix);ce,se=ct,st
        module=profile['loader']=='HuggingFace';n_layers=model.config.num_hidden_layers if module else model.cfg.n_layers
        nodes=[]
        for layer in range(n_layers):
            if module:
                for basis in ['resid_post','mlp_out','attn_out']:nodes.append(DriverNode(basis=basis,layer=layer,position='ANSWER'))
            else:
                for pos in (['END','IO','S1','S2'] if config['experiment_id']=='ioi_nominal' else ['ANSWER']):
                    for h in range(model.cfg.n_heads):nodes.append(DriverNode(basis='head_pos',layer=layer,position=pos,component=h))
        if smoke:nodes=nodes[-min(8,len(nodes)):]
        k=1 if smoke else config['max_support'];ridge=config['refit']['lambda']
        disc=DriverCircuitDiscovery(adapter=adapter,target_metric=target,candidate_nodes=nodes,epsilon=.1 if config['experiment_id']=='ioi_nominal' else .2,max_drivers=k,ridge=ridge,lin_tol=1e-3,fallback_mode='actual_gain',max_prune_queries=64)
        methods=['driver_omp'] if smoke else ['driver_omp','coverage_driver','top_attribution','random_matched','actual_gain_greedy']
        for m in methods:
            start=time.time();calls={'driver_omp':lambda:disc.run(ct,st),'coverage_driver':lambda:coverage_driver_baseline(disc,ct,st,k=k,coverage_target=.95),'top_attribution':lambda:top_node_attribution_baseline(disc,ct,st,k=k),'random_matched':lambda:random_matched_baseline(disc,ct,st,k=k,seed=seed),'actual_gain_greedy':lambda:actual_gain_greedy_baseline(disc,ct,st,k=k,topk_proposal=32)}
            result=calls[m]();nodeset=result.drivers
            with torch.no_grad():
                c=adapter.cache_activations(ce,nodeset);s=adapter.cache_activations(se,nodeset);patch={n:float(a)*(c[n]-s[n]) for n,a in zip(nodeset,result.strengths)}
                t=adapter.run_clean_or_corrupt(ce,'clean',target);b=adapter.run_clean_or_corrupt(se,'corrupt',target);y=adapter.run_with_patches(se,patch,target)
            rows.append(dict(experiment_id=config['experiment_id'],seed=seed,method=m,smoke=smoke,k=k,candidate_count=len(nodes),drivers=[n.label() for n in nodeset],coefficients=native(result.strengths),source_behavior=native(b),target_behavior=native(t),forward_behavior=native(y),Rel_tar=rel_tar(native(b),native(t),native(y)),R_clean=r_clean(native(t),native(y)),elapsed_seconds=time.time()-start,fit_target=serial(ct),fit_source=serial(st)))
    (Path(output)/'measurements.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    return dict(records=len(rows),model_execution=True,selection=True,coefficient_refit=True,smoke=smoke,loaded_model=model._sic_execution_metadata)
