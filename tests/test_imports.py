import subprocess,sys

def test_light_import_and_help(root):
 s="import sparse_control,sys; assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
 subprocess.run([sys.executable,'-c',s],check=True,cwd=root)
 subprocess.run([sys.executable,'-m','sparse_control','--help'],check=True,cwd=root,capture_output=True)
def test_registered_producer_imports(root):
 import pytest,importlib
 pytest.importorskip('torch')
 from sparse_control.io import load_json
 for p in (root/'configs').glob('*.json'):
  x=load_json(p)
  if x.get('full_run'):importlib.import_module(x['full_run']['module'])
