import copy
import json
import sys
import types
import pytest


@pytest.mark.parametrize('loader', ['TransformerLens','HuggingFace'])
def test_loader_receives_profile_and_records_actuals(root, monkeypatch, tmp_path, loader):
    torch = pytest.importorskip('torch')
    from sparse_control.runners import model_loading as m
    profiles = json.loads((root/'experiments/execution_profiles.json').read_text())
    p = copy.deepcopy(profiles['mps_head_fp32' if loader == 'TransformerLens' else 'mps_module_bf16'])
    p['backend'] = 'cpu'
    revision = 'a'*40
    calls = []
    tok = types.SimpleNamespace(padding_side='left', truncation_side='left', pad_token=None, eos_token='eos')
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.w = torch.nn.Parameter(torch.ones(1,dtype=getattr(torch,p['dtype'])))
            self.tokenizer = tok; self.config = types.SimpleNamespace(_commit_hash=revision)
    class Factory:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(kwargs); return Model()
    if loader == 'TransformerLens':
        monkeypatch.setitem(sys.modules,'transformer_lens',types.SimpleNamespace(HookedTransformer=Factory))
    else:
        tokenizer_factory = types.SimpleNamespace(from_pretrained=lambda *a,**k: tok)
        monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoModelForCausalLM=Factory,AutoTokenizer=tokenizer_factory))
    monkeypatch.setenv('SIC_EXECUTION_PROFILE', json.dumps(p))
    destination = tmp_path/'loads.jsonl'
    monkeypatch.setenv('SIC_MODEL_METADATA',str(destination))
    if loader == 'TransformerLens': model = m.load_tl('test',device='cpu',revision=revision)
    else:
        tokenizer = m.load_tokenizer('test',revision=revision)
        model = m.load_hf('test',device='cpu',revision=revision,tokenizer=tokenizer)
    actual = json.loads(destination.read_text())
    assert actual['dtype'] == p['dtype'] and actual['revision_observed'] == revision
    assert tok.padding_side == tok.truncation_side == 'right'
    assert calls[0]['revision'] == revision
    if loader == 'TransformerLens':
        assert calls[0]['dtype'] == torch.float32
        for k,v in p['model_loader_options'].items(): assert calls[0][k] == v
    else:
        assert calls[0]['attn_implementation'] == 'eager'
        assert calls[0]['torch_dtype'] == torch.bfloat16
    bad = dict(p,dtype='float64')
    with pytest.raises(ValueError, match='dtype/device'):
        m.observe(model,tok,'test',revision,bad,{})


def test_unsupported_interventions_not_ignored(root):
    from sparse_control.runners.dispatch import execute
    with pytest.raises(ValueError, match='interventions'):
        execute(root,'ioi_nominal','full','0','mps_head_fp32',None,interventions='unused.jsonl',plan=True)
