"""Configured model comparison experiment."""
from .dispatch import execute
def run_model_comparison(root,experiment,profile,seeds,output):
    return execute(root,experiment,"full",seeds,profile,output)
