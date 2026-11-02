"""Configured recurrence experiment."""
from .dispatch import execute
def run_recurrence(root,experiment,profile,seeds,output):
    return execute(root,experiment,"full",seeds,profile,output)
