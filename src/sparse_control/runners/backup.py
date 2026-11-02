"""Configured backup experiment."""
from .dispatch import execute
def run_backup(root,experiment,profile,seeds,output):
    return execute(root,experiment,"full",seeds,profile,output)
