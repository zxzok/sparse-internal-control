"""Full-vocabulary baseline-first KL, natural logarithms, nearest-rank P95."""
import math,statistics
def kl_unpatched_patched(unpatched_logits,patched_logits):
    import numpy as np
    a,b=np.asarray(unpatched_logits,dtype=np.float64),np.asarray(patched_logits,dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("Finite matching logit arrays required")
    if a.ndim==0 or a.shape[-1]<2:raise ValueError("Full vocabulary axis required")
    def logsoftmax(x):
        x=x-x.max(axis=-1,keepdims=True);return x-np.log(np.exp(x).sum(axis=-1,keepdims=True))
    p,q=logsoftmax(a),logsoftmax(b);return (np.exp(p)*(p-q)).sum(axis=-1)
def ioi_position_delta_kl(patched,corrupt,clean):return kl_unpatched_patched(patched,clean)-kl_unpatched_patched(corrupt,clean)
def preservation_summary(values,seed_ids,threshold=.001):
    if len(values)!=len(seed_ids) or not values:raise ValueError("KL and seed lengths differ")
    if any(not math.isfinite(x) or x < -1e-9 for x in values):raise ValueError("Invalid KL")
    by={s:[x for x,t in zip(values,seed_ids) if t==s] for s in set(seed_ids)}
    v=sorted(values)
    return dict(mean_kl=statistics.fmean(v),median_kl=statistics.median(v),p95_kl=v[math.ceil(.95*len(v))-1],max_kl=max(v),prompt_passes=sum(x<=threshold for x in v),total=len(v),seed_total=len(by),mean_kl_seed_passes=sum(statistics.fmean(x)<=threshold for x in by.values()),all_prompt_seed_passes=sum(all(v<=threshold for v in x) for x in by.values()))
