"""Configured sentiment fixed experiment."""
from .dispatch import execute
def run_sentiment_fixed(root,experiment,profile,seeds,output):
    return execute(root,experiment,"full",seeds,profile,output)
