"""Command-line entry points for record replay and explicit model execution."""
import argparse,json,sys
from pathlib import Path
from .registry import package_root,load_artifacts,resolve_dependencies
from .io import output_directory,write_json

def reproduce(root,output,artifacts='all'):
 from .reporting.tables import build_table
 ar=load_artifacts(root);ids=list(ar) if artifacts=='all' else artifacts.split(',');done=[]
 for aid in resolve_dependencies(ar,ids):
  build_table(aid,root,output)
  done.append(aid)
 write_json(Path(output)/'replay.json',dict(artifacts=done,scope='tabular_saved_records',model_execution=False));return done

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
 for name in ['reproduce','verify','run','check-release','build-release']:
  q=sp.add_parser(name);q.add_argument('--root',type=Path)
  if name in ['reproduce','verify']:q.add_argument('--artifacts',default='all');q.add_argument('--output',type=Path,required=True)
  if name=='run':
   q.add_argument('--experiment',required=True);q.add_argument('--stage',required=True,choices=['full','fit','nominal','dose','reversal','feedback','preservation','generation']);q.add_argument('--profile',required=True);q.add_argument('--seeds');q.add_argument('--output',type=Path);q.add_argument('--interventions',type=Path);q.add_argument('--smoke',action='store_true');q.add_argument('--plan',action='store_true');q.add_argument('--task',choices=['mcqa_humanities','mcqa_stem','sentiment','refusal'])
  if name=='check-release':q.add_argument('--manifest',action='store_true')
  if name=='build-release':q.add_argument('--output',type=Path,required=True)
 a=p.parse_args(argv)
 try:
  root=package_root(a.root)
  if a.command=='reproduce':out=output_directory(a.output,root);result={'artifacts':reproduce(root,out,a.artifacts),'scope':'tabular_saved_records','model_execution':False}
  elif a.command=='verify':
   from .reporting.verify import verify_outputs
   result=verify_outputs(root,a.output,a.artifacts)
  elif a.command=='run':
   from .runners.dispatch import execute
   if not a.plan and a.output is None:raise ValueError('--output required for execution')
   result=execute(root,a.experiment,a.stage,a.seeds,a.profile,a.output,a.interventions,a.smoke,a.plan,a.task)
  elif a.command=='check-release':
   from .release.check import check_release
   result=check_release(root,manifest=a.manifest)
  else:
   from .release.build import build_release
   result=build_release(root,a.output)
  print(json.dumps(result,indent=2,ensure_ascii=False,default=str,allow_nan=False));return 0
 except (ValueError,RuntimeError,FileNotFoundError,FileExistsError,ImportError) as exc:
  print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
