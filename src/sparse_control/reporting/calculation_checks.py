"""Arithmetic checks for the existing verify command, independent of table generators."""
from __future__ import annotations
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METHODS = ('driver_omp', 'coverage_driver', 'top_attribution')
ALL_METHODS = METHODS + ('random_matched', 'actual_gain_greedy')
SERIES = {
    'ioi_nominal_same_panel': ('ioi', 10, 64, ALL_METHODS, ('P',)),
    'mcqa_humanities_module_option_logits_same_panel': ('mcqa_humanities', 3, 32, ALL_METHODS, ('P',)),
    'mcqa_stem_module_option_logits_same_panel': ('mcqa_stem', 3, 32, ALL_METHODS, ('P',)),
    'mcqa_humanities_head_scalar_heldout': ('mcqa_humanities', 8, 12, METHODS, ('P', 'F')),
    'mcqa_stem_head_scalar_heldout': ('mcqa_stem', 8, 12, METHODS, ('P', 'F')),
    'sentiment_paired_heldout': ('sentiment', 8, 8, METHODS, ('P', 'F')),
    'refusal_head_scalar_heldout': ('refusal', 8, 10, METHODS, ('P', 'F')),
}
EPSILONS = (0.30, 0.35, 0.40, 0.45, 0.50)
GROUP_FIELDS = ('series', 'task', 'method', 'direction')

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def finite_vector(value: object, label: str) -> list[float]:
    require(isinstance(value, list) and bool(value), f'{label}: nonempty list required')
    out = [float(x) for x in value]
    require(all(math.isfinite(x) for x in out), f'{label}: nonfinite value')
    return out

def norm(value: list[float]) -> float:
    return math.sqrt(math.fsum(x*x for x in value))

def distance(a: list[float], b: list[float]) -> float:
    require(len(a) == len(b), 'Mismatched vector dimensions')
    return norm([x-y for x, y in zip(a, b)])

def key(row: dict, fields: tuple[str, ...]) -> tuple:
    return tuple(row[x] for x in fields)

def read_csv(path: Path, key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        k = key(row, key_fields)
        require(k not in result, f'{path.name}: duplicate key {k}')
        result[k] = row
    return result

def verify(root: Path, result_root: Path, tolerance: float) -> dict:
    require(tolerance > 0 and math.isfinite(tolerance), 'Positive finite tolerance required')
    path = root/'data/measurements/formal_reachability/readouts.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(rows) == 272, f'Expected 272 vectors, found {len(rows)}')
    records, grouped, record_ids = {}, defaultdict(list), set()
    for row in rows:
        name = row['series']
        require(name in SERIES, f'Unknown series: {name}')
        task, nseeds, length, methods, modes = SERIES[name]
        require(row['task'] == task and row['method'] in methods and row['direction'] in modes, 'Unexpected condition')
        seed = row['seed']
        require(isinstance(seed, int) and 0 <= seed < nseeds, 'Unexpected seed')
        require(row['dose'] == 1, 'Reachability requires nominal dose one')
        rid = row['record_id']
        require(rid not in record_ids, f'Duplicate record_id: {rid}')
        record_ids.add(rid)
        s, t, y = [finite_vector(row[field], field) for field in ('source_behavior','target_behavior','forward_behavior')]
        require(len(s) == len(t) == len(y) == length, f'{rid}: incorrect vector length')
        require(len(row['drivers']) == len(row['coefficients']), f'{rid}: support/coefficient mismatch')
        residual, displacement, target_norm = distance(t,y), distance(t,s), norm(t)
        calculated = dict(residual_norm=residual, displacement_norm=displacement, target_norm=target_norm,
                          Rel_tar=residual/(displacement+1e-6), R_clean=residual/max(target_norm,1e-12),
                          unpatched_Rel_tar=displacement/(displacement+1e-6))
        group = key(row, GROUP_FIELDS)
        identity = group+(str(seed),)
        require(identity not in records, f'Duplicate seed condition: {identity}')
        records[identity] = calculated
        grouped[group].append((seed, calculated))
    expected_groups = {(series,task,method,mode) for series,(task,_,_,methods,modes) in SERIES.items() for method in methods for mode in modes}
    require(set(grouped) == expected_groups, 'Missing or extra condition groups')
    for group, values in grouped.items():
        require(sorted(seed for seed,_ in values) == list(range(SERIES[group[0]][1])), f'Incomplete seeds: {group}')
    folder = result_root/'formal_reachability'
    per = read_csv(folder/'per_seed.csv', GROUP_FIELDS+('seed',))
    summary = read_csv(folder/'summary.csv', GROUP_FIELDS)
    sens_raw = read_csv(folder/'threshold_sensitivity.csv', GROUP_FIELDS+('epsilon',))
    sens = {}
    for k, v in sens_raw.items():
        normalized = k[:-1]+(float(k[-1]),)
        require(normalized not in sens, 'Duplicate numerical threshold key')
        sens[normalized] = v
    require(set(per) == set(records), 'Per-seed CSV coverage mismatch')
    require(set(summary) == expected_groups, 'Summary CSV coverage mismatch')
    require(set(sens) == {g+(e,) for g in expected_groups for e in EPSILONS}, 'Threshold CSV coverage mismatch')
    max_difference = 0.0
    def close(observed: object, expected: float, label: str) -> None:
        nonlocal max_difference
        observed = float(observed)
        require(math.isfinite(observed), f'{label}: nonfinite CSV value')
        delta = abs(observed-expected)
        max_difference = max(max_difference, delta)
        require(delta <= tolerance, f'{label}: {observed} != {expected}')
    for identity, values in records.items():
        for field, value in values.items(): close(per[identity][field], value, f'per_seed {identity} {field}')
        require(per[identity]['passes_0p4'].lower() in ('true','false'), 'Invalid Boolean')
        require((per[identity]['passes_0p4'].lower() == 'true') == (values['Rel_tar'] <= .4), 'Per-seed pass mismatch')
    for group, raw in grouped.items():
        values = [v for _,v in raw]
        rel = [v['Rel_tar'] for v in values]; clean = [v['R_clean'] for v in values]
        d = summary[group]
        for field,value in {'Rel_tar_mean':statistics.mean(rel),'Rel_tar_sample_sd':statistics.stdev(rel),
                            'R_clean_mean':statistics.mean(clean),'R_clean_sample_sd':statistics.stdev(clean)}.items():
            close(d[field],value,f'summary {group} {field}')
        require(int(d['seeds']) == len(raw), 'Summary seed count mismatch')
        require(int(d['passes_0p4']) == sum(v <= .4 for v in rel), 'Summary pass count mismatch')
        for epsilon in EPSILONS:
            d=sens[group+(epsilon,)]
            require(int(d['seeds']) == len(raw), 'Threshold seed count mismatch')
            require(int(d['passes']) == sum(v <= epsilon for v in rel), f'Threshold pass count mismatch: {group}, {epsilon}')
    return dict(status='PASS', seed_vectors=len(records), condition_groups=len(grouped), threshold_rows=len(sens),
                maximum_absolute_difference=max_difference, model_execution=False)

