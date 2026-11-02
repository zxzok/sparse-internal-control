import csv
import json
import shutil
import subprocess
import sys
import pytest
from sparse_control.cli import reproduce
from sparse_control.reporting.verify import verify_outputs


@pytest.fixture(scope='module')
def replay(root, tmp_path_factory):
    folder = tmp_path_factory.mktemp('replay')
    reproduce(root, folder)
    return folder


def independent(root, output):
    code = ('from sparse_control.reporting.calculation_checks import verify; '
            'from pathlib import Path; import json; '
            f'print(json.dumps(verify(Path({str(root)!r}), Path({str(output)!r}), 1e-12)))')
    return subprocess.run([sys.executable, '-S', '-c', 'import sys;sys.path.insert(0,'+repr(str(root/'src'))+');'+code],
                          capture_output=True, text=True)


def test_independent_standard_library(root, replay):
    result = independent(root, replay)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert (report['seed_vectors'], report['condition_groups'], report['threshold_rows']) == (272,39,195)
    r = verify_outputs(root, replay)
    assert r['scope'] == 'tabular_saved_records'
    assert r['paper_values']['numeric_rows'] == 161


@pytest.mark.parametrize('damage', ['missing_table', 'bad_threshold', 'missing_seed', 'bad_summary', 'bad_statistics', 'bad_tex'])
def test_damaged_outputs_fail(root, replay, tmp_path, damage):
    output = tmp_path/'damaged'
    shutil.copytree(replay, output)
    if damage == 'missing_table':
        (output/'tables/nominal_reconstruction.csv').unlink()
    elif damage in ['bad_threshold', 'missing_seed', 'bad_summary']:
        name = dict(bad_threshold='threshold_sensitivity', missing_seed='per_seed', bad_summary='summary')[damage]
        p = output/'formal_reachability'/f'{name}.csv'
        with p.open(newline='') as stream:
            reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
        if damage == 'bad_threshold': rows[0]['passes'] = '999'
        elif damage == 'bad_summary': rows[0]['Rel_tar_mean'] = '999'
        else: rows.pop()
        with p.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader();writer.writerows(rows)
        assert independent(root, output).returncode != 0
    else:
        name = 'nominal_reconstruction_statistics.csv' if damage == 'bad_statistics' else 'nominal_reconstruction.tex'
        (output/'tables'/name).write_text('corrupted\n')
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_outputs(root, output)
    if damage == 'bad_threshold':
        from sparse_control.cli import main
        assert main(['verify','--root',str(root),'--output',str(output)]) != 0


def test_subset_requires_explicit_scope(root, tmp_path):
    reproduce(root, tmp_path, 'nominal_reconstruction')
    with pytest.raises(ValueError, match='scope'):
        verify_outputs(root, tmp_path)
    report = verify_outputs(root, tmp_path, 'nominal_reconstruction')
    assert report['artifacts'] == ['nominal_reconstruction']
    assert report['formal_reachability']['status'] == 'not_in_requested_scope'
