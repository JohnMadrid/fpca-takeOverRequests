import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already = any("GROUP ED: BEFORE vs AFTER OPERATION C" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source",""))
              for c in nb["cells"])
assert not already, "group-C-compare cell already present."

# insert after the coordination sig-test cell (fallback: after trial)
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "COORDINATION SIG TESTING" in "".join(c["source"]): ins=i
if ins is None:
    for i,c in enumerate(nb["cells"]):
        if c["cell_type"]=="code" and "COORDINATION TRIAL" in "".join(c["source"]): ins=i
assert ins is not None

code = r'''# ============================================================
# GROUP ED: BEFORE vs AFTER OPERATION C  (post window)
# ------------------------------------------------------------
# Success vs Failure (stratified-balanced by ExperimentalCondition) and
# Base vs Full (equal n), in the POST window, reported with operation C OFF
# (legacy MFPCA) and C ON (gold standard: per-driver temporal demeaning).
# Equal N within each comparison; permutation p (group-label shuffle, stratified
# for S vs F). Self-contained; reads car_reference CSVs.
# ============================================================
import numpy as np, os
import dask.dataframe as dd
from numpy.linalg import eigh as _eighG
from tqdm.auto import tqdm

_GC_PERM = 1000          # perms (raise for final)
_GC_R    = 100           # equal-n subsample reps for the observed/null stat
_GC_RNG  = np.random.default_rng(0)
_GC_MODS = ["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_GC_M    = len(_GC_MODS)
_GC_EVENTS = ["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_GC_LAB  = {"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_GC_POST = np.arange(250,450)
_GC_DIR  = os.getcwd()+"/data/cleaned_data/data_segment/car_reference/"

def _gc_ed(jev):
    jev=np.maximum(jev,0); s=jev.sum()
    return float((s**2)/np.sum(jev**2)) if s>0 else np.nan

def _gc_selfstd(a):
    f=a.reshape(-1,a.shape[2]); mu=f.mean(0); sd=f.std(0,ddof=1); sd[sd<1e-12]=1.0
    return (a-mu)/sd

def _gc_twostep(a, use_C):
    """MFPCA two-step ED on (W,T,M). use_C toggles per-driver temporal demeaning."""
    a=a.astype(float)
    if use_C:
        a=a - a.mean(axis=1, keepdims=True)     # OPERATION C
    a=_gc_selfstd(a)
    stacked=[]
    for ci in range(_GC_M):
        Xc=a[:,:,ci]-a[:,:,ci].mean(0)
        ev,evec=_eighG(np.cov(Xc.T)); o=np.argsort(ev)[::-1]; ev=np.maximum(ev[o],0); evec=evec[:,o]
        nk=max(1,int(np.sum(ev>1e-12)))
        stacked.append(Xc@evec[:,:nk])
    Z=np.hstack(stacked); Z=Z-Z.mean(0)
    return _gc_ed(np.maximum(_eighG(np.cov(Z.T))[0],0))

def _gc_load(evt):
    df=dd.read_csv(_GC_DIR+f"car_reference_{evt}.csv", assume_missing=True,
                   blocksize="100MB", dtype={'HitObjectName':'object'}).compute()
    X=[]; succ=[]; cond=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        sub=sub.sort_values('time_from_event')
        X.append(sub[_GC_MODS].values.astype(float)[_GC_POST,:])
        succ.append(float(sub['SuccessfulCompletionState'].iloc[0]))
        cond.append(str(sub['ExperimentalCondition'].iloc[0]))
    return np.stack(X,0), (np.array(succ)==1.0), np.array(cond)

# equal-n machinery (strata = condition for S/F; constant for Base/Full)
def _alloc(strata, iA, iB, target):
    levs=np.unique(strata)
    cap={s:min(int(np.sum(strata[iA]==s)),int(np.sum(strata[iB]==s))) for s in levs}
    tot=sum(cap.values())
    if tot<target: return None
    raw={s:target*cap[s]/tot for s in levs}; base={s:int(np.floor(raw[s])) for s in levs}
    rem=target-sum(base.values()); order=sorted(levs,key=lambda s:raw[s]-base[s],reverse=True); i=0
    while rem>0 and i<10*len(levs):
        s=order[i%len(order)]
        if base[s]<cap[s]: base[s]+=1; rem-=1
        i+=1
    return base
def _equal_n(mA,mB,strata):
    iA=np.where(mA)[0]; iB=np.where(mB)[0]
    return sum(min(int(np.sum(strata[iA]==s)),int(np.sum(strata[iB]==s))) for s in np.unique(strata))
def _draw(mA,mB,strata,rng,target):
    iA=np.where(mA)[0]; iB=np.where(mB)[0]; al=_alloc(strata,iA,iB,target)
    if al is None: return None,None
    sA=[];sB=[]
    for s,k in al.items():
        if k==0: continue
        a=iA[strata[iA]==s]; b=iB[strata[iB]==s]
        sA.append(rng.choice(a,k,replace=False)); sB.append(rng.choice(b,k,replace=False))
    return np.concatenate(sA),np.concatenate(sB)
def _permlabels(mA,mB,strata,rng):
    N=len(strata); pA=np.zeros(N,bool); pB=np.zeros(N,bool)
    iA=np.where(mA)[0]; iB=np.where(mB)[0]
    for s in np.unique(strata):
        pool=np.concatenate([iA[strata[iA]==s],iB[strata[iB]==s]])
        if len(pool)==0: continue
        nA=int(np.sum(strata[iA]==s)); pm=rng.permutation(pool)
        pA[pm[:nA]]=True; pB[pm[nA:]]=True
    return pA,pB

def _grp(arr,mA,mB,strata,N,use_C,rng,R=_GC_R):
    a=np.empty(R); b=np.empty(R)
    for r in range(R):
        sA,sB=_draw(mA,mB,strata,rng,N)
        a[r]=_gc_twostep(arr[sA],use_C); b[r]=_gc_twostep(arr[sB],use_C)
    return float(np.mean(a)),float(np.mean(b))
def _grp_p(arr,mA,mB,strata,N,use_C,rng,P=_GC_PERM):
    gA,gB=_grp(arr,mA,mB,strata,N,use_C,rng); obs=gA-gB
    null=np.empty(P)
    for p in range(P):
        pA,pB=_permlabels(mA,mB,strata,rng); sA,sB=_draw(pA,pB,strata,rng,N)
        null[p]=_gc_twostep(arr[sA],use_C)-_gc_twostep(arr[sB],use_C)
    return gA,gB,obs,float((np.sum(np.abs(null)>=abs(obs))+1)/(P+1))

print("GROUP ED (post window): BEFORE vs AFTER operation C")
print("="*92)
print(f"{'event':<14}{'comparison':<20}{'N':>4}  {'A(C off)':>9}{'B(C off)':>9}{'p_off':>7}   {'A(C on)':>9}{'B(C on)':>9}{'p_on':>7}")
print("-"*92)
for _e in tqdm(_GC_EVENTS, desc="events"):
    arr,succ,cond=_gc_load(_e)
    comps=[("Success vs Failure",succ,~succ,cond),
           ("Base vs Full",(cond=='BaseCondition'),(cond=='FullLoopAR'),np.zeros(len(cond),int))]
    for cname,mA,mB,strata in comps:
        N=_equal_n(mA,mB,strata)
        if N<5:
            print(f"{_GC_LAB[_e]:<14}{cname:<20}{'low n':>4}"); continue
        aA0,aB0,_,p0=_grp_p(arr,mA,mB,strata,N,False,_GC_RNG)
        aA1,aB1,_,p1=_grp_p(arr,mA,mB,strata,N,True ,_GC_RNG)
        print(f"{_GC_LAB[_e]:<14}{cname:<20}{N:>4}  {aA0:>9.2f}{aB0:>9.2f}{p0:>7.3f}   {aA1:>9.2f}{aB1:>9.2f}{p1:>7.3f}")
    print("-"*92)
print("\nA = first group (Success / Base), B = second (Failure / Full). N = equal n per arm.")
print("C off = legacy MFPCA; C on = gold standard (per-driver temporal demeaning).")
print(f"Perms={_GC_PERM}, equal-n reps={_GC_R}. Raise _GC_PERM for final numbers.")
'''
ast.parse(code)
nb["cells"].insert(ins+1, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted group before/after-C compare cell at {ins+1}. Total cells: {len(nb['cells'])}.")
