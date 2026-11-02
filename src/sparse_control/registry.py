"""Experiment and artifact dependency resolution."""
from pathlib import Path
from .io import load_json, resolve_path
def package_root(root=None):
    if root is not None: return Path(root).resolve()
    import os
    if os.environ.get("SPARSE_CONTROL_ROOT"): return Path(os.environ["SPARSE_CONTROL_ROOT"]).resolve()
    for p in (Path.cwd(), *Path(__file__).resolve().parents):
        if (p/"experiments/registry.json").exists(): return p
    raise ValueError("Use --root or SPARSE_CONTROL_ROOT to locate experimental data")
def load_experiments(root=None):
    root=package_root(root); records=load_json(root/"experiments/registry.json")
    return {r["experiment_id"]:r for r in records}
def load_artifacts(root=None):
    root=package_root(root); records=load_json(root/"experiments/artifacts.json")
    return {r["artifact_id"]:r for r in records}
def resolve_dependencies(registry, requested):
    ordered=[]; active=set(); done=set()
    def visit(k):
        if k in active: raise ValueError(f"Dependency cycle: {k}")
        if k in done:return
        if k not in registry:raise ValueError(f"Unknown dependency: {k}")
        active.add(k)
        for d in registry[k].get("dependencies",[]):visit(d)
        active.remove(k);done.add(k);ordered.append(k)
    for k in requested:visit(k)
    return ordered
def validate_registry(root=None):
    root=package_root(root); ex=load_experiments(root); ar=load_artifacts(root)
    resolve_dependencies(ar,list(ar))
    for k,r in ar.items():
        if not r.get("generator") or not r.get("record_paths"): raise ValueError(f"Incomplete artifact: {k}")
        for e in r["experiment_ids"]:
            if e not in ex: raise ValueError(f"{k}: unknown experiment {e}")
        for p in r["record_paths"]:
            if not resolve_path(root,p).is_file():raise FileNotFoundError(f"{k}: {p}")
    return {"experiments":len(ex),"artifacts":len(ar)}
