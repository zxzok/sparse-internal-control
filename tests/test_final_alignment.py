"""Paper alignment checks and saved-record sensitivity, without model execution."""
import json
import shutil
import pytest
from sparse_control.runners.dispatch import execute
from sparse_control.reporting.tables import ioi_feedback, sentiment_displacement_sensitivity
from sparse_control.reporting.paper_checks import check_paper_references
from sparse_control.cli import reproduce


def test_ioi_feedback_details_from_terminal_readouts(root):
    row=ioi_feedback(root)
    assert (row['hits'],row['total'],row['initial_hits'])==(24,24,2)
    assert row['mean_absolute_error']==pytest.approx(0.13513108963767687)
    assert row['mean_evaluations']==pytest.approx(95/24)


def test_sentiment_nonzero_displacement_sensitivity(root):
    per,summary=sentiment_displacement_sensitivity(root)
    assert len(per)==48 and len(summary)==6
    excluded=[2,3,2,2,3,3,2,3]
    for r in per:assert r['excluded_pairs']==excluded[r['seed']]
    expected={('driver_omp','P'):(.44536362305581484,.08795794162284805,3),
              ('driver_omp','F'):(.721555789850777,.09202389023051499,0),
              ('coverage_driver','P'):(.17133654876375687,.08381840040204128,8),
              ('coverage_driver','F'):(.6078932661500506,.111443926552793,0),
              ('top_attribution','P'):(.16656840954892632,.08354053562532868,8),
              ('top_attribution','F'):(.6070533484968667,.11237759560398866,0)}
    for r in summary:
        mean,sd,hits=expected[r['method'],r['direction']]
        assert r['Rel_tar_nonzero_mean']==pytest.approx(mean,abs=1e-12)
        assert r['Rel_tar_nonzero_sample_sd']==pytest.approx(sd,abs=1e-12)
        assert r['passes_all']==r['passes_nonzero']==hits


@pytest.mark.parametrize('experiment,profile',[('ioi_nominal','mps_head_fp32'),('mcqa_humanities_nominal','mps_module_bf16'),('mcqa_stem_nominal','mps_module_bf16')])
def test_smoke_plan_one_seed_and_nominal_label(root,experiment,profile):
    plan=execute(root,experiment,'nominal',None,profile,None,smoke=True,plan=True)
    assert plan['seeds']==[0] and plan['stage']=='nominal' and plan['smoke']


@pytest.mark.parametrize('experiment,stage,seeds',[
    ('ioi_nominal','feedback',None),('ioi_nominal','nominal','0,1'),
    ('ioi_feedback','nominal','0'),('sentiment_paired','nominal','0')])
def test_invalid_smoke_rejected_before_output(root,tmp_path,experiment,stage,seeds):
    out=tmp_path/'run'
    with pytest.raises(ValueError,match='Smoke requires'):
        execute(root,experiment,stage,seeds,'mps_head_fp32',out,smoke=True)
    assert not out.exists()


def test_non_smoke_seed_scope_preserved(root):
    plan=execute(root,'ioi_nominal','nominal',None,'mps_head_fp32',None,plan=True)
    assert plan['seeds']==list(range(10))


@pytest.mark.parametrize('damage',['config_and_output','tex','registry','transcription'])
def test_stale_static_tables_and_references_rejected(root,tmp_path,damage):
    copy=tmp_path/'source';copy.mkdir();out=tmp_path/'replay';out.mkdir()
    for name in ['experiments','configs','results']:shutil.copytree(root/name,copy/name)
    reproduce(root,out,'control_axes,experiment_matrix')
    assert check_paper_references(copy,out,['control_axes','experiment_matrix'])['static_tables']==2
    if damage=='config_and_output':
        for p in [copy/'configs/display_control_axes.json',out/'tables/control_axes.json']:
            d=json.loads(p.read_text());d['cells'][0][2]=r'$\mathrm{Rel}_{\mathrm{tar}}(\gamma)$';p.write_text(json.dumps(d))
    elif damage=='tex':
        p=out/'tables/experiment_matrix.tex';p.write_text(p.read_text().replace('64 fitting/64 test pairs', 'fitting/test split'))
    elif damage=='registry':
        p=copy/'experiments/artifacts.json';d=json.loads(p.read_text());d[0]['paper_display_name']='Table 99';p.write_text(json.dumps(d))
    else:
        p=copy/'results/paper_values.json';d=json.loads(p.read_text());d['tables']['nominal_reconstruction']['paper_display_name']='Table 2';p.write_text(json.dumps(d))
    ids=['nominal_reconstruction'] if damage=='transcription' else ['control_axes','experiment_matrix']
    with pytest.raises(ValueError):check_paper_references(copy,out,ids)
