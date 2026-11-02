"""Verify a declared tabular output scope and independently check paper calculations."""
import json
import tempfile
from pathlib import Path
from ..io import load_json
from ..registry import load_artifacts, resolve_dependencies
from ..release.check import check_records
from .paper_checks import compare_with_expected


def verify_outputs(root, output, artifacts='all'):
    root, output = Path(root), Path(output)
    registry = load_artifacts(root)
    requested = list(registry) if artifacts == 'all' else artifacts.split(',')
    selected = resolve_dependencies(registry, requested)
    manifest = load_json(output/'replay.json')
    if manifest != dict(artifacts=selected, scope='tabular_saved_records', model_execution=False):
        raise ValueError('Output scope differs from requested artifacts; specify --artifacts for a subset')
    record_check = check_records(root)
    independent = compare_with_expected(output, root/'results/paper_values.json', selected)
    formal = {'status': 'not_in_requested_scope'}
    if any(x.startswith('formal_reachability_') or x == 'reachability_tolerance' for x in selected):
        from .calculation_checks import verify as verify_calculations
        formal = verify_calculations(root, output, 1e-12)
    # This regeneration is an integrity check, separate from independent arithmetic.
    from ..cli import reproduce
    with tempfile.TemporaryDirectory(prefix='sic-check-') as folder:
        reference = Path(folder)
        reproduce(root, reference, artifacts)
        paths = sorted(p.relative_to(reference) for p in reference.rglob('*') if p.is_file())
        declared = {Path(path) for aid in selected for path in registry[aid]['output_files']}
        if not declared <= set(paths):
            raise ValueError(f'Registered outputs were not generated: {sorted(map(str, declared-set(paths)))}')
        for relative in paths:
            actual, expected = output/relative, reference/relative
            if not actual.is_file():
                raise FileNotFoundError(f'Missing registered output: {relative}')
            same = (load_json(actual) == load_json(expected)) if relative.suffix == '.json' else (actual.read_bytes() == expected.read_bytes())
            if not same:
                raise ValueError(f'Generated output differs from saved-record calculation: {relative}')
        actual_paths = {p.relative_to(output) for p in output.rglob('*') if p.is_file()}
        # The distributed results also contain the independent paper transcription.
        extras = actual_paths-set(paths)-{Path('paper_values.json')}
        if extras:
            raise ValueError(f'Unexpected files outside declared output scope: {sorted(map(str, extras))}')
    return dict(status='PASS', scope='tabular_saved_records', artifacts=selected,
                output_files=len(paths), paper_values=independent, formal_reachability=formal,
                records=record_check, preservation='saved_KL_scalar_aggregation_only',
                model_execution=False)
