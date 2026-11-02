"""Complete-vector target and clean-output-normalized errors."""
import math
def vector(x):
    if hasattr(x,"reshape"):x=x.reshape(-1).tolist()
    def flatten(a):
        for v in a:
            if isinstance(v,(list,tuple)):yield from flatten(v)
            else:yield float(v)
    out=list(flatten(x))
    if not out or not all(math.isfinite(v) for v in out):raise ValueError("Expected a nonempty finite vector")
    return out
def norm(x):return math.sqrt(math.fsum(v*v for v in vector(x)))
def distance(a,b):
    a,b=vector(a),vector(b)
    if len(a)!=len(b):raise ValueError("Vector dimension mismatch")
    return math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b)))
def rel_tar(source,target,patched,xi=1e-6):
    if not math.isfinite(xi) or xi<=0: raise ValueError("xi must be positive and finite")
    return distance(target,patched)/(distance(target,source)+xi)
def r_clean(target,patched):return distance(target,patched)/max(norm(target),1e-12)
