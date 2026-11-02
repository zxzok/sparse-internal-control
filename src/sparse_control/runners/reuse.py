"""Configured reuse experiment."""
from .dispatch import execute
def run_reuse(root,experiment,profile,seeds,output):
    return execute(root,experiment,"full",seeds,profile,output)
