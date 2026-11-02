"""Series-specific recovery statistics, distinct from 1-Rel_tar."""
import statistics
def mcqa_recovery(source,target,patched):
    s,t,p=map(statistics.fmean,(source,target,patched));d=t-s
    return None if abs(d)<1e-9 else (p-s)/d
def sentiment_recovery(source,target,patched):
    s,t,p=map(statistics.fmean,(source,target,patched));return (p-s)/max(t-s,1e-6)
def ioi_recovery(target,patched,floor=1e-6,failure_threshold=.3):
    if len(target)!=len(patched) or not target:raise ValueError("Matching prompt readouts required")
    errors=[abs(t-p)/max(abs(t),floor) for t,p in zip(target,patched)]
    return dict(recovery=1-statistics.fmean(errors),failure_fraction=sum(e>failure_threshold for e in errors)/len(errors),pointwise_errors=errors)
