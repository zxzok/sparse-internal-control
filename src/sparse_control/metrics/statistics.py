"""Seed summaries preserve statistical units and threshold precision."""
import math,statistics
def seed_summary(values, seeds=None, expected_seeds=None):
    v=list(map(float,values))
    if not v or not all(math.isfinite(x) for x in v):raise ValueError("Nonempty finite values required")
    if seeds is not None:
        if len(seeds)!=len(v) or len(set(seeds))!=len(seeds):raise ValueError("Duplicate/mismatched seeds")
        if expected_seeds is not None and set(seeds)!=set(expected_seeds):raise ValueError("Missing or unexpected seed")
    return dict(mean=statistics.fmean(v),sample_sd=statistics.stdev(v) if len(v)>1 else None,n=len(v),minimum=min(v),maximum=max(v))
def threshold_counts(values,threshold):return sum(float(x)<=threshold for x in values)
def format_mean_sd(values,digits=3):
    s=seed_summary(values)
    if s['sample_sd'] is None:raise ValueError("Sample SD requires two seeds")
    return f"{s['mean']:.{digits}f} ± {s['sample_sd']:.{digits}f}"
