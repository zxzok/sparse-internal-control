"""Feedback endpoints are terminal evaluations, not best trajectory values."""
import statistics
def endpoint_metrics(target,terminal_margin,initial_margin,tolerance,evaluations=None):
    error=terminal_margin-target
    return dict(error=error,absolute_error=abs(error),hit=abs(error)<=tolerance,initial_hit=abs(initial_margin-target)<=tolerance,evaluations=evaluations)
def summarize_feedback(records):
    if not records:raise ValueError("Missing endpoints")
    return dict(hits=sum(r['hit'] for r in records),total=len(records),initial_hits=sum(r['initial_hit'] for r in records),mean_absolute_error=statistics.fmean(r['absolute_error'] for r in records),mean_evaluations=statistics.fmean(r['evaluations'] for r in records) if all(r['evaluations'] is not None for r in records) else None)
def update_dose(gamma,target,current,gain):return gamma+gain*(target-current)
