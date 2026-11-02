"""Build only explicitly allowed files and emit an external ZIP checksum."""
import hashlib,zipfile
from pathlib import Path
from .check import check_release,load_allowlist
from ..io import resolve_path

def build_release(root,output):
 root,output=Path(root).resolve(),Path(output).resolve()
 if output.is_relative_to(root):raise ValueError('Archive must be outside the release directory')
 check_release(root,manifest=True);files=load_allowlist(root);output.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for rel in sorted(files):z.write(resolve_path(root,rel),'sparse-internal-control/'+rel)
 digest=hashlib.sha256(output.read_bytes()).hexdigest()
 with Path(str(output)+'.sha256').open('x') as f:f.write(digest+'  '+output.name+'\n')
 return dict(archive=output.name,bytes=output.stat().st_size,files=len(files),sha256=digest)
