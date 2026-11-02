"""Bind fixed-control evaluation to saved ordered inputs and baseline readouts."""
import hashlib
import json
import math
from pathlib import Path
from ..io import load_json, load_jsonl


def plain(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def compare_fields(actual, expected, label='input'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'{label}: saved input fields differ')
        for key in expected:
            compare_fields(actual[key], expected[key], f'{label}.{key}')
    elif actual != expected:
        raise ValueError(f'{label}: saved input differs')


def check_saved_inputs(root, experiment, seed, batches):
    ct, st, ce, se = map(plain, batches)
    path = Path(root)/'data/inputs'/experiment/'batches.jsonl'
    if not path.exists():
        return dict(status='reconstructed_without_saved_tokens', seed=seed)
    rows = [r for r in load_jsonl(path) if r['seed'] == seed]
    if len(rows) != 1:
        raise ValueError(f'{experiment}: missing or duplicate saved batch for seed {seed}')
    saved = rows[0]
    if 'clean' in saved:
        pairs = [(ct, saved['clean']), (st, saved['source']),
                 (ce, saved['clean']), (se, saved['source'])]
    else:
        pairs = list(zip([ct, st, ce, se], [saved[k] for k in
                     ('train_target', 'train_source', 'test_target', 'test_source')]))
    for name, (actual, expected) in zip(('fit_target', 'fit_source', 'evaluation_target', 'evaluation_source'), pairs):
        compare_fields(actual, expected, f'{experiment}.seed{seed}.{name}')
    return dict(status='exact_saved_tokens_masks_positions_readout_ids', seed=seed,
                saved_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def check_baselines(source, target, references, seed, atol=1e-4):
    rows = [r for r in references if r['seed'] == seed and 'source_behavior' in r and 'target_behavior' in r]
    if not rows:
        return dict(status='no_saved_baseline_vectors', seed=seed)
    maximum = 0.0
    for row in rows:
        for actual, field in ((source, 'source_behavior'), (target, 'target_behavior')):
            expected = row[field]
            if len(actual) != len(expected):
                raise ValueError(f'{field}: baseline dimension mismatch')
            if not all(math.isfinite(x) for x in actual+expected):
                raise ValueError(f'{field}: nonfinite baseline')
            delta = max(abs(a-b) for a, b in zip(actual, expected))
            maximum = max(maximum, delta)
            if delta > atol:
                raise ValueError(f'{field}: seed {seed} baseline error {delta:.9g} exceeds {atol}')
    return dict(status='PASS', seed=seed, vector_comparisons=2*len(rows),
                maximum_absolute_error=maximum, atol=atol, rtol=0)


def check_recorded_model(root, config, revision):
    path = Path(root)/'data/measurements'/config['experiment_id']/'run_metadata.json'
    if not path.exists():
        return 'original_revision_not_recorded'
    recorded = load_json(path)
    for key, current in [('model', config['model_asset_id']), ('revision', revision)]:
        if recorded.get(key) and recorded[key] != current:
            raise ValueError(f'Saved {key} does not match selected model asset')
    return 'saved_model_revision_matches' if recorded.get('revision') else 'original_revision_not_recorded'
