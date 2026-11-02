"""Strict, portable reading and non-destructive result writing."""
import csv, json
from pathlib import Path

def resolve_path(root, relative):
    root=Path(root).resolve(); p=Path(relative)
    if p.is_absolute() or ".." in p.parts: raise ValueError(f"Expected package-relative path: {relative}")
    result=(root/p).resolve()
    if not result.is_relative_to(root): raise ValueError(f"Path escapes package: {relative}")
    return result

def _constant(s): raise ValueError(f"Non-standard JSON number: {s}")
def load_json(path):
    with Path(path).open(encoding="utf8") as f: return json.load(f,parse_constant=_constant)
def load_jsonl(path):
    rows=[]
    with Path(path).open(encoding="utf8") as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            try: rows.append(json.loads(line,parse_constant=_constant))
            except (ValueError,TypeError) as exc: raise ValueError(f"{Path(path).name}: line {i}: {exc}") from exc
    return rows
def load_arrays(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def write_csv(path, rows):
    rows=list(rows)
    if not rows: raise ValueError("Cannot write an empty measurement table")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f: json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write("\n")
def output_directory(path, root=None):
    p=Path(path).resolve()
    if root:
        root=Path(root).resolve()
        for d in ("data","results","src","configs","experiments","assets","tests","scripts","docs","requirements"):
            if p.is_relative_to(root/d): raise ValueError("Output cannot overwrite distributed inputs or expected results")
    if p.exists() and any(p.iterdir()): raise FileExistsError(f"Output directory is not empty: {p.name}")
    p.mkdir(parents=True,exist_ok=True);return p
