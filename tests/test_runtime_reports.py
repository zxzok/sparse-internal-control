"""Report propagation with fake numerical backends; no model or weights loaded."""
import copy
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self): return self
    def float(self): return self
    def double(self): return self
    def reshape(self, *shape): return self
    def numpy(self): return self


@pytest.mark.parametrize('experiment,stage,task', [
    ('preservation', 'preservation', None),
    ('preservation', 'preservation', 'refusal'),
    ('refusal_single_site', 'preservation', None),
    ('refusal_generation', 'generation', None),
])
def test_observed_model_metadata_reaches_saved_report(root, tmp_path, monkeypatch,
                                                     experiment, stage, task):
    from sparse_control.runners import dispatch, preservation, refusal_generation
    from sparse_control.metrics import preservation as metrics

    profile_id = 'mps_head_fp32'
    profile = json.loads((root/'experiments/execution_profiles.json').read_text())[profile_id]
    monkeypatch.setattr(dispatch.importlib.metadata, 'version', lambda name: profile['versions'][name])
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        no_grad=nullcontext, tensor=lambda values, **kwargs: FakeTensor(values),
        equal=lambda left, right: left == right))
    monkeypatch.setitem(sys.modules, 'numpy', SimpleNamespace(savez_compressed=lambda *a, **kw: None))
    batch = {'tokens': [[1, 2]], 'answer_pos': [1]}
    monkeypatch.setitem(sys.modules, 'sparse_control.tasks.pairs', SimpleNamespace(
        MCQAExample=lambda *args: None, build_mcqa_pairs=lambda *a, **kw: (batch, batch)))
    monkeypatch.setattr(metrics, 'kl_unpatched_patched', lambda *args: 0.001)
    observed = []

    def load_model(_root, config, _profile):
        metadata = dict(model_id=config['model_asset_id'], loader='test_loader',
                        dtype=config['dtype'], device='test_backend',
                        revision_requested='a' * 40, revision_observed='a' * 40,
                        tokenizer_padding_side='right', tokenizer_truncation_side='right',
                        loader_options_applied={'test_load_index': len(observed)})
        observed.append((config['experiment_id'], copy.deepcopy(metadata)))
        adapter = SimpleNamespace(
            run_clean_or_corrupt=lambda *args: FakeTensor([1., 2.]),
            run_with_patches=lambda *args: FakeTensor([1., 2.]))
        return SimpleNamespace(_sic_execution_metadata=metadata), None, adapter

    record = dict(seed=0, method='driver_omp', drivers=[], alphas=[],
                  prompt_id='test', prompt_text='Test input',
                  off_companion_correct=0, off_companion_distractor=1)
    for runner in [preservation, refusal_generation]:
        monkeypatch.setattr(runner, 'load_model', load_model)
        monkeypatch.setattr(runner, 'load_jsonl', lambda path: [record])
        monkeypatch.setattr(runner, 'build_panel', lambda *args: (batch, batch, batch, batch, None))
        monkeypatch.setattr(runner, 'contrast', lambda *args, **kwargs: {})
    original_load_json = preservation.load_json

    def load_json(path):
        if path.name == 'panel.json':
            return {'records': [dict(prompt_id='test', prompt='Test input',
                                     question='Test?', choices=['a', 'b', 'c', 'd'])]}
        return original_load_json(path)

    monkeypatch.setattr(preservation, 'load_json', load_json)
    monkeypatch.setattr(preservation, 'unpadded', lambda *args: batch)
    monkeypatch.setattr(preservation, 'serial', lambda value: value)
    monkeypatch.setattr(refusal_generation, '_generate_single',
                        lambda *args: {'generated_token_ids': [1] * 16})

    output = tmp_path/'run'
    report = dispatch.execute(root, experiment, stage, '0', profile_id, output, task=task)
    assert json.loads((output/'run_metadata.json').read_text()) == report
    assert report['status'] == 'COMPLETE'
    assert report['selection'] is False and report['coefficient_refit'] is False
    if stage == 'generation':
        assert len(observed) == report['records'] == 1
        assert report['loaded_model'] == observed[0][1]
        assert len(json.loads((output/'generation.jsonl').read_text())['generated_token_ids']) == 16
    else:
        tasks = ['mcqa_humanities', 'mcqa_stem', 'sentiment', 'refusal'] if experiment == 'preservation' and task is None else ['refusal']
        assert len(observed) == report['records'] == len(tasks)
        assert report['loaded_models'] == [
            dict(task=name, source_experiment_id=eid, **metadata)
            for name, (eid, metadata) in zip(tasks, observed)]
        for name in tasks:
            assert json.loads((output/name/'prompt_kl.jsonl').read_text())['kl'] == 0.001
