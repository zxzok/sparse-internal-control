"""Compare generated cells to independently transcribed final-paper values."""
import collections,json,re
from pathlib import Path
from ..io import load_json

def tex_table_cells(text):
    """Read the simple three-column static tables, including longtable headers."""
    header=text.split(r'\toprule',1)[1].split(r'\midrule',1)[0]
    body=text.rsplit(r'\endhead',1)[-1] if r'\endhead' in text else text.split(r'\midrule',1)[1]
    body=body.split(r'\bottomrule',1)[0]
    def rows(s):
        return [[re.sub(r'\s+',' ',c).strip() for c in r.split('&')] for r in s.split(r'\\') if r.strip()]
    return rows(header)[0],rows(body)

def check_paper_references(root,output,artifact_ids=None):
    """Check table identity and static definitions against manuscript transcription."""
    root,output=Path(root),Path(output)
    reference=load_json(root/'experiments/paper_reference.json')
    registry={r['artifact_id']:r for r in load_json(root/'experiments/artifacts.json')}
    paper=load_json(root/'results/paper_values.json')['tables']
    selected=list(registry) if artifact_ids is None else artifact_ids
    static_count=0
    for aid in selected:
        if aid not in reference['artifact_references']:raise ValueError(f'Missing paper reference: {aid}')
        for field,value in reference['artifact_references'][aid].items():
            if registry[aid].get(field)!=value:raise ValueError(f'{aid}: registry paper reference mismatch: {field}')
            if aid in paper and paper[aid].get(field)!=value:raise ValueError(f'{aid}: transcribed paper reference mismatch: {field}')
        if aid not in reference['tables']:continue
        expected=reference['tables'][aid];config=load_json(root/'configs'/f'display_{aid}.json')
        actual=load_json(output/'tables'/f'{aid}.json')
        for name,spec in [('configuration',config),('output',actual)]:
            if spec['columns']!=expected['columns'] or spec['cells']!=expected['cells']:
                raise ValueError(f'{aid}: static table {name} differs from manuscript')
        for text in [config['latex_table'],(output/'tables'/f'{aid}.tex').read_text()]:
            if tex_table_cells(text)!=(expected['columns'],expected['cells']):
                raise ValueError(f'{aid}: static LaTeX table differs from manuscript')
        static_count+=1
    return dict(status='PASS',artifact_references=len(selected),static_tables=static_count)
def canonical(x):
    x=str(x).replace(r'\DriverOMP{}','Driver-OMP').replace(r'\CoverageDriver{}','Coverage-Driver').replace(r'\DriverOMP','Driver-OMP').replace(r'\CoverageDriver','Coverage-Driver')
    x=x.replace(r'\pm','±').replace(r'\(','').replace(r'\)','').replace('$','').replace('−','-')
    x=x.replace('Evaluation-pair (P)','P').replace('Training-mean (F)','F')
    return re.sub(r'\s+','',x).replace('\\','')
def normalized(rows):
    out=[];last=None
    for row in rows:
        r=list(map(canonical,row))
        if r and r[0]=='':r[0]=last
        if r:last=r[0]
        out.append(tuple(r))
    return collections.Counter(out)
def compare_with_expected(output,expected,artifact_ids=None):
    output=Path(output);expected=load_json(expected);checks=[]
    for aid,e in expected['tables'].items():
        if artifact_ids is not None and aid not in artifact_ids:continue
        p=output/'tables'/(aid+'.json')
        if not p.exists():raise FileNotFoundError(f'Missing generated table: {aid}')
        actual=load_json(p)['cells']
        if normalized(actual)!=normalized(e['cells']):
            missing=normalized(e['cells'])-normalized(actual);extra=normalized(actual)-normalized(e['cells'])
            raise ValueError(f'{aid}: paper cells mismatch; missing={list(missing)[:2]}, unexpected={list(extra)[:2]}')
        checks.append(dict(artifact_id=aid,rows=len(actual)))
    supporting_count=0
    supporting=load_json(output/'supporting_numbers.json') if artifact_ids is None or 'supporting_numbers' in artifact_ids else None
    for key,e in (expected['supporting_numbers'].items() if supporting is not None else []):
        value=supporting
        for part in key.split('.'):value=value[part]
        if 'digits' in e:ok=round(value,e['digits'])==e['value']
        else:ok=value==e['value']
        if not ok:raise ValueError(f'Supporting number mismatch: {key}')
        supporting_count+=1
    return dict(status='PASS',scope='independently_transcribed_paper_values',tables=len(checks),numeric_rows=sum(x['rows'] for x in checks),checks=checks,supporting_numbers=supporting_count,model_execution=False)
