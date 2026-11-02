import copy
import json
import pytest
from sparse_control.runners.input_checks import check_saved_inputs, check_baselines, check_recorded_model


def saved_batches(row):
    if 'clean' in row:
        return [row['clean'], row['source'], row['clean'], row['source']]
    return [row[k] for k in ['train_target','train_source','test_target','test_source']]


def test_all_saved_batch_schemas(root):
    n = 0
    for p in (root/'data/inputs').glob('*/batches.jsonl'):
        for row in map(json.loads, p.read_text().splitlines()):
            report = check_saved_inputs(root, p.parent.name, row['seed'], saved_batches(row))
            assert report['status'].startswith('exact_saved_')
            n += 1
    assert n == 32


@pytest.mark.parametrize('field', ['tokens','attention_mask','answer_pos','correct_indices','choice_token_ids','positions'])
def test_changed_inputs_rejected(root, field):
    row = json.loads((root/'data/inputs/mcqa_humanities_head/batches.jsonl').read_text().splitlines()[0])
    batches = copy.deepcopy(saved_batches(row))
    value = batches[2][field]
    if isinstance(value, dict): value = value['ANSWER']
    if isinstance(value[0], list): value[0][0] += 1
    else: value[0] += 1
    with pytest.raises(ValueError, match='saved input differs'):
        check_saved_inputs(root, 'mcqa_humanities_head', 0, batches)


def test_missing_and_mismatched_baselines():
    r = [dict(seed=0, source_behavior=[1,2], target_behavior=[3,4])]
    assert check_baselines([1,2], [3,4], r, 0)['maximum_absolute_error'] == 0
    with pytest.raises(ValueError, match='exceeds'):
        check_baselines([1,2.001], [3,4], r, 0)
    with pytest.raises(ValueError, match='dimension'):
        check_baselines([1], [3,4], r, 0)
    with pytest.raises(ValueError, match='nonfinite'):
        check_baselines([1,float('nan')], [3,4], r, 0)
    assert check_baselines([1],[2],[],0)['status'] == 'no_saved_baseline_vectors'


def test_wrong_revision_rejected(root):
    with pytest.raises(ValueError, match='revision'):
        check_recorded_model(root, {'experiment_id':'ioi_nominal','model_asset_id':'gpt2'}, 'wrong')
