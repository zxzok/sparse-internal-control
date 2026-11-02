"""Four-method Qwen3-8B comparison with the residual-only library."""
import argparse,json,time
from pathlib import Path
from .qwen8b_setup import build_qwen35_9b_mcqa
from .nominal_context import resolve_device,set_determinism,summarize_result
from ...selection.core import DriverCircuitDiscovery
from ...selection.baselines import coverage_driver_baseline,top_node_attribution_baseline,random_matched_baseline
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',default='Qwen/Qwen3-8B',choices=['Qwen/Qwen3-8B']);p.add_argument('--device',required=True);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--seeds',required=True);p.add_argument('--ks',default='8',choices=['8']);p.add_argument('--epsilon',type=float,default=.2);p.add_argument('--ridge',type=float,default=1.);p.add_argument('--mcqa-source',default='mmlu_humanities');p.add_argument('--eap-edges',type=int,default=24);p.add_argument('--baselines',default='coverage_driver,top_attribution,random_matched',choices=['coverage_driver,top_attribution,random_matched']);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();a.with_mlp=False;device=resolve_device(a.device)
 with a.out.open('x') as f:
  for seed in map(int,a.seeds.split(',')):
   a.seed=seed;set_determinism(seed);adapter,target,nodes,clean,corrupt=build_qwen35_9b_mcqa(a,device)
   d=DriverCircuitDiscovery(adapter=adapter,target_metric=target,candidate_nodes=nodes,epsilon=a.epsilon,max_drivers=8,ridge=a.ridge)
   calls={'driver_omp':lambda:d.run(clean,corrupt),'coverage_driver':lambda:coverage_driver_baseline(d,clean,corrupt,k=8),'top_attribution':lambda:top_node_attribution_baseline(d,clean,corrupt,k=8),'random_matched':lambda:random_matched_baseline(d,clean,corrupt,k=8,seed=seed)}
   for name,call in calls.items():
    start=time.time();r=call();row=summarize_result(name,seed,8,r,time.time()-start);row['ridge']=a.ridge;f.write(json.dumps(row)+'\n');f.flush()
if __name__=='__main__':main()
