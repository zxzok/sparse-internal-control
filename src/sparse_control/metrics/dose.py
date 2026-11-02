"""Signed fixed-grid dose-response statistics."""
import math,statistics
from .reachability import vector
def ranks(x):
    order=sorted(range(len(x)),key=lambda i:x[i]);out=[0.]*len(x);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and x[order[j]]==x[order[i]]:j+=1
        for k in range(i,j):out[order[k]]=(i+j-1)/2+1
        i=j
    return out
def signed_spearman(x,y):
    x,y=vector(x),vector(y)
    if len(x)!=len(y) or len(x)<2:raise ValueError("Paired dose vectors required")
    a,b=ranks(x),ranks(y);ma,mb=statistics.fmean(a),statistics.fmean(b)
    den=math.sqrt(math.fsum((v-ma)**2 for v in a)*math.fsum((v-mb)**2 for v in b))
    return math.fsum((v-ma)*(w-mb) for v,w in zip(a,b))/den if den else None
def vector_progress(source,target,patched,xi=1e-6):
    s,t,p=map(vector,(source,target,patched))
    if len(s)!=len(t) or len(t)!=len(p):raise ValueError("Vector dimension mismatch")
    q=[b-a for a,b in zip(s,t)];f=[b-a for a,b in zip(s,p)]
    return math.fsum(a*b for a,b in zip(q,f))/(math.fsum(a*a for a in q)+xi)
def mean_progress(source_mean,target_mean,patched_mean,denominator_floor=None):
    d=target_mean-source_mean
    if denominator_floor is not None:d=max(d,denominator_floor)
    if d==0:raise ValueError("Undefined zero mean displacement")
    return (patched_mean-source_mean)/d
