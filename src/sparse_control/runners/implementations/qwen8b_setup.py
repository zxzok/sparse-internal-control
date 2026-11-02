"""
Task-specific executed experiment implementation.
"""
from __future__ import annotations
from sparse_control.runners.model_loading import load_tl, load_hf, load_tokenizer
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import torch
from sparse_control.runners.implementations.nominal_context import resolve_device, set_determinism, summarize_result
from sparse_control.selection.core import DriverCircuitDiscovery, DriverNode
from sparse_control.selection.execution_context import coverage_driver_baseline, eap_endpoint_nodes, eap_ig_endpoint_nodes, random_matched_baseline, top_node_attribution_baseline

def build_qwen35_9b_mcqa(args, device: str):
    """Qwen3.5-9B MCQA preset.

    Returns (adapter, target_metric, candidate_nodes, clean_batch, corrupt_batch).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparse_control.adapters.huggingface import HFHookAdapter
    from sparse_control.adapters.transformerlens import MCQALogitVector
    from sparse_control.tasks.pairs import build_mcqa_pairs, load_mmlu_examples
    print(f'[preset] loading {args.model} (this takes a few min on first run)...', flush=True)
    t0 = time.time()
    tokenizer = load_tokenizer(args.model, trust_remote_code=True, revision=os.environ['SIC_MODEL_REVISION'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device != 'cpu' else torch.float32
    model = load_hf(args.model, device=device, tokenizer=tokenizer, torch_dtype=dtype, attn_implementation='eager', trust_remote_code=True, low_cpu_mem_usage=True, revision=os.environ['SIC_MODEL_REVISION'])
    model.train(False)
    print(f'[preset] loaded in {time.time() - t0:.1f}s', flush=True)
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        if hasattr(model.model.language_model, 'model'):
            layers_root_path = 'model.language_model.model.layers'
        else:
            layers_root_path = 'model.language_model.layers'
        text_config = model.config.text_config
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers_root_path = 'model.layers'
        text_config = getattr(model.config, 'text_config', model.config)
    else:
        raise SystemExit('Could not resolve Qwen3.5 layer path. Inspect model structure via `print(model)` and update `build_qwen35_9b_mcqa`.')
    n_layers = int(text_config.num_hidden_layers)
    layer_types = list(getattr(text_config, 'layer_types', ['full_attention'] * n_layers))
    print(f'[preset] layers root: {layers_root_path}, n={n_layers}', flush=True)
    print(f'[preset] layer_types: {layer_types}', flush=True)
    adapter = HFHookAdapter(model, tokenizer, device=device, layers_attr=layers_root_path, resid_module=lambda l: f'{layers_root_path}.{l}', mlp_module=lambda l: f'{layers_root_path}.{l}.mlp', attn_module=lambda l: f'{layers_root_path}.{l}.self_attn')
    examples = load_mmlu_examples(category=args.mcqa_source.replace('mmlu_', ''), n=args.batch_size, seed=args.seed)
    clean, corrupt = build_mcqa_pairs(tokenizer, examples, device=device)
    target = MCQALogitVector()
    nodes: List[DriverNode] = [DriverNode(basis='resid_post', layer=layer, position='ANSWER') for layer in range(n_layers)]
    if args.with_mlp:
        nodes += [DriverNode(basis='mlp_out', layer=layer, position='ANSWER') for layer in range(n_layers)]
    print(f"[preset] {len(nodes)} candidate nodes ({('resid+mlp' if args.with_mlp else 'resid only')})", flush=True)
    return (adapter, target, nodes, clean, corrupt)
