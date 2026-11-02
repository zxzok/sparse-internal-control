"""Apply explicit execution profiles at each model-loading call."""
import json
import os
from pathlib import Path


def resolve_profile(profile=None, loader=None, device=None):
    p = profile if profile is not None else json.loads(os.environ['SIC_EXECUTION_PROFILE'])
    if loader and p['loader'] != loader:
        raise ValueError('Execution profile loader does not match producer')
    if device and p['backend'] != device:
        raise ValueError('Execution profile device does not match producer')
    return p


def configure_tokenizer(tokenizer, profile):
    tokenizer.padding_side = profile['tokenizer_padding_side']
    tokenizer.truncation_side = profile['tokenizer_truncation_side']
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def observe(model, tokenizer, model_id, revision, profile, options):
    parameter = next(model.parameters())
    dtype = str(parameter.dtype).replace('torch.', '')
    device = parameter.device.type
    if dtype != profile['dtype'] or device != profile['backend']:
        raise ValueError(f'Actual model dtype/device {dtype}/{device} differs from profile')
    if tokenizer.padding_side != profile['tokenizer_padding_side'] or tokenizer.truncation_side != profile['tokenizer_truncation_side']:
        raise ValueError('Actual tokenizer sides differ from profile')
    cfg = getattr(model, 'config', None)
    observed_revision = getattr(cfg, '_commit_hash', None)
    if observed_revision is not None and observed_revision != revision:
        raise ValueError('Loaded checkpoint revision differs from requested revision')
    metadata = dict(model_id=model_id, revision_requested=revision,
                    revision_observed=observed_revision or 'not_recorded_by_loader',
                    loader=profile['loader'], dtype=dtype, device=device,
                    tokenizer_padding_side=tokenizer.padding_side,
                    tokenizer_truncation_side=tokenizer.truncation_side,
                    loader_options_applied=options)
    model._sic_execution_metadata = metadata
    if os.environ.get('SIC_MODEL_METADATA'):
        with Path(os.environ['SIC_MODEL_METADATA']).open('a') as stream:
            stream.write(json.dumps(metadata, default=str)+'\n')
    return metadata


def load_tl(model_id, *, device, revision, profile=None):
    import torch
    from transformer_lens import HookedTransformer
    p = resolve_profile(profile, 'TransformerLens', device)
    options = dict(p['model_loader_options'])
    model = HookedTransformer.from_pretrained(model_id, device=device, revision=revision,
                                               dtype=getattr(torch, p['dtype']), **options)
    configure_tokenizer(model.tokenizer, p)
    observe(model, model.tokenizer, model_id, revision, p, options)
    return model


def load_tokenizer(model_id, *, revision, profile=None, **options):
    from transformers import AutoTokenizer
    p = resolve_profile(profile, 'HuggingFace')
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, **options)
    configure_tokenizer(tokenizer, p)
    return tokenizer


def load_hf(model_id, *, device, revision, tokenizer, profile=None, **native_options):
    import torch
    from transformers import AutoModelForCausalLM
    p = resolve_profile(profile, 'HuggingFace', device)
    options = dict(native_options)
    wanted = dict(torch_dtype=getattr(torch, p['dtype']), attn_implementation=p['attention_implementation'])
    for key, value in wanted.items():
        if key in options and options[key] != value:
            raise ValueError(f'Native producer {key} differs from profile')
        options[key] = value
    for key, value in p.get('model_loader_options', {}).items():
        key = {'attention_implementation': 'attn_implementation'}.get(key, key)
        value = getattr(torch, value) if key == 'torch_dtype' else value
        if key in options and options[key] != value:
            raise ValueError(f'Profile loader option conflicts with producer: {key}')
        options[key] = value
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **options).to(device)
    configure_tokenizer(tokenizer, p)
    observe(model, tokenizer, model_id, revision, p, {k: str(v) if k == 'torch_dtype' else v for k, v in options.items()})
    return model
