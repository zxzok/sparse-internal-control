"""Target-side negative-patch distances; direction is not flipped twice."""
from .reachability import distance
def reversal_metrics(source,target,reversed_behavior,xi=1e-6,threshold=2):
    d=distance(target,source)+xi; ds=distance(reversed_behavior,source);dt=distance(reversed_behavior,target)
    status="undefined" if ds==0 and dt==0 else "positive_infinity" if ds==0 else "finite"
    ratio=dt/ds if ds else None
    return dict(source_distance=ds,target_distance=dt,relative_source=ds/d,relative_target=dt/d,ratio=ratio,status=status,passes=(ratio>=threshold) if status=='finite' else (True if status=='positive_infinity' else None))
