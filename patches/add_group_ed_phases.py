import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already = any("GROUP ED PHASES" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source",""))
              for c in nb["cells"])
assert not already, "group ED phases cell already present."

# insert after the region-perm cell
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "REGION-LABEL PERMUTATION TEST" in "".join(c["source"]): ins=i
assert ins is not None, "region-perm cell not found"

md = r'''## Group differences in the rise-and-fall (collapse) effect

Two hypotheses, each tested WITHIN every event (n = drivers per arm, well powered),
with the 3 events as independent replications (report k/3 significant; no pooling).

- **H1 Success vs Failure** (stratified by warning condition so both arms have the
  same warning mix): is the action-phase collapse stronger in one outcome group?
- **H2 Visual (HUD+FullLoopAR) vs Non-visual (Base+Audio)** (balanced by outcome so
  both arms have equal success rate): is the collapse weaker for visual warnings?

Group ED(t) is recomputed per group at equal balanced n (subsample, average over
draws). Effect metric (safest first): **collapse depth = Baseline - Action ED**
(the robust limb). Group difference tested by label-shuffle (shuffle group labels
within strata, recompute both group EDs, recompute the difference).
Gold recipe (C + pooled), outliers-removed car-frame, mean-RT phase boundary.
'''

code = r'''# ============================================================
# GROUP ED PHASES: collapse-depth difference between groups (per event)
# ------------------------------------------------------------
# Effect metric = Baseline - Action ED (collapse below baseline; robust limb).
# Group ED(t) recomputed at equal balanced n. Difference tested within each event
# by stratified group-label shuffle. Events = independent replications (k/3).
# H1: Success vs Failure (strat by condition). H2: Visual vs Non-visual (bal by outcome).
# ============================================================
import numpy as np, os
import dask.dataframe as dd
from numpy.linalg import eigh as _eighGE
from tqdm.auto import tqdm

_GE_PERM = 2000
_GE_R    = 100          # subsample draws to stabilise a group's ED curve
_GE_RNG  = np.random.default_rng(0)
_GE_VARS = ["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_GE_M    = len(_GE_VARS)
_GE_EVENTS=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_GE_LAB={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_GE_DIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/outliers_removed_5features/"

# reuse mean RT per event from gold cell if present, else compute
try:
    _GE_RT={e:_mean_rt_evt(e) for e in _GE_EVENTS}
except NameError:
    import pandas as pd
    _rt=pd.read_csv(os.getcwd()+"/reaction_times.csv")
    _rt["RT_Steering"]=pd.to_numeric(_rt["RT_Steering"],errors="coerce")
    _rt=_rt[(_rt["Unavailable_Steering"].astype(str).str.lower()!="true") & _rt["RT_Steering"].notna()]
    _GE_RT={e:float(_rt[_rt["EventName"]==e]["RT_Steering"].mean()) for e in _GE_EVENTS}

def _ge_load(evt):
    """RAW per-driver 8s curves + labels. Returns X(W,T,M), secs, succ(bool), cond(str)."""
    df=dd.read_csv(_GE_DIR+f"car_reference_{evt}.csv",assume_missing=True,blocksize="100MB",
                   dtype={'HitObjectName':'object'}).compute()
    def fv(s): s=s.dropna(); return s.iloc[0] if len(s) else np.nan
    X=[]; suc=[]; cond=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        sub=sub.sort_values('time_from_event')
        raw=sub[_GE_VARS].values.astype(float)
        s0=np.linspace(-5,5,raw.shape[0]); raw=raw[(s0>=-4)&(s0<=4)]
        X.append(raw); suc.append(float(fv(sub['SuccessfulCompletionState'])))
        cond.append(str(fv(sub['ExperimentalCondition'])))
    secs=np.linspace(-5,5,points_per_window); secs=secs[(secs>=-4)&(secs<=4)]
    return np.stack(X,0), secs, (np.array(suc)==1.0), np.array(cond)

def _ge_ed_curve(arr):
    """Gold ED(t): per-driver center (C) + pooled per-channel scale."""
    A=arr.astype(float)-arr.astype(float).mean(axis=1,keepdims=True)
    flat=A.reshape(-1,_GE_M); psd=flat.std(0,ddof=1); psd[psd<1e-12]=1.0
    W,T,_=A.shape; ed=np.zeros(T)
    for t in range(T):
        snap=A[:,t,:].copy(); snap=snap-snap.mean(0); snap=snap/psd
        ev=np.maximum(_eighGE(snap.T@snap/(W-1),eigvals_only=True),0); s=ev.sum()
        ed[t]=(s**2)/np.sum(ev**2) if s>0 else np.nan
    return ed

def _collapse_depth(arr, secs, rt):
    """Baseline - Action ED on this group's curve."""
    ed=_ge_ed_curve(arr)
    base=(secs>=-4)&(secs<0); act=(secs>=rt)&(secs<=4)
    return float(ed[base].mean()-ed[act].mean())

# stratified equal-n draw (strata array; constant for H2-by-outcome handled by caller)
def _alloc(strata,iA,iB,target):
    levs=np.unique(strata)
    cap={s:min(int((strata[iA]==s).sum()),int((strata[iB]==s).sum())) for s in levs}
    tot=sum(cap.values())
    if tot<target: target=tot
    raw={s:target*cap[s]/tot for s in levs}; base={s:int(np.floor(raw[s])) for s in levs}
    rem=target-sum(base.values()); order=sorted(levs,key=lambda s:raw[s]-base[s],reverse=True); k=0
    while rem>0 and k<10*len(levs):
        s=order[k%len(order)]
        if base[s]<cap[s]: base[s]+=1; rem-=1
        k+=1
    return base,target
def _draw(strata,iA,iB,rng,target):
    al,_=_alloc(strata,iA,iB,target); sA=[];sB=[]
    for s,kk in al.items():
        if kk==0: continue
        a=iA[strata[iA]==s]; b=iB[strata[iB]==s]
        sA.append(rng.choice(a,kk,replace=False)); sB.append(rng.choice(b,kk,replace=False))
    return np.concatenate(sA),np.concatenate(sB)
def _equal_n(strata,iA,iB):
    return sum(min(int((strata[iA]==s).sum()),int((strata[iB]==s).sum())) for s in np.unique(strata))
def _permlabels(strata,iA,iB,rng):
    N=len(strata); pA=np.zeros(N,bool); pB=np.zeros(N,bool)
    for s in np.unique(strata):
        pool=np.concatenate([iA[strata[iA]==s],iB[strata[iB]==s]])
        if len(pool)==0: continue
        nA=int((strata[iA]==s).sum()); pm=rng.permutation(pool)
        pA[pm[:nA]]=True; pB[pm[nA:]]=True
    return pA,pB

def _group_collapse(arr,secs,rt,mA,mB,strata,N,rng,R=_GE_R):
    """Mean collapse-depth for each arm over R balanced draws, and their diff."""
    iA=np.where(mA)[0]; iB=np.where(mB)[0]
    cA=np.empty(R); cB=np.empty(R)
    for r in range(R):
        sA,sB=_draw(strata,iA,iB,rng,N)
        cA[r]=_collapse_depth(arr[sA],secs,rt); cB[r]=_collapse_depth(arr[sB],secs,rt)
    return float(cA.mean()),float(cB.mean())

def _test(arr,secs,rt,mA,mB,strata,P=_GE_PERM,rng=None):
    rng=rng or _GE_RNG
    N=_equal_n(strata,np.where(mA)[0],np.where(mB)[0])
    if N<5: return None
    cA,cB=_group_collapse(arr,secs,rt,mA,mB,strata,N,rng); obs=cA-cB
    null=np.empty(P)
    for p in range(P):
        pA,pB=_permlabels(strata,np.where(mA)[0],np.where(mB)[0],rng)
        sA,sB=_draw(strata,np.where(pA)[0],np.where(pB)[0],rng,N)
        null[p]=_collapse_depth(arr[sA],secs,rt)-_collapse_depth(arr[sB],secs,rt)
    pv=(np.sum(np.abs(null)>=abs(obs))+1)/(P+1)
    return dict(N=N,cA=cA,cB=cB,diff=obs,p=float(pv))

def _star(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""

# ---- run both hypotheses per event ----
print("GROUP ED PHASES -- collapse depth (Baseline - Action), per-event tests")
print("collapse>0 = ED below baseline in action phase. diff = group A - group B.")
print("="*92)
_ge_res={"H1":[],"H2":[]}
for _e in tqdm(_GE_EVENTS, desc="events"):
    arr,secs,succ,cond=_ge_load(_e); rt=_GE_RT[_e]
    # H1: Success vs Failure, strat by condition
    rH1=_test(arr,secs,rt,succ,~succ,cond)
    # H2: Visual(HUD+Full) vs Non-visual(Base+Audio), balanced by outcome (strata=outcome)
    vis=np.isin(cond,["HUDOnly","FullLoopAR"]); non=np.isin(cond,["BaseCondition","AudioOnly"])
    strata_out=succ.astype(int)  # balance by success/failure
    rH2=_test(arr,secs,rt,vis,non,strata_out)
    _ge_res["H1"].append((_e,rH1)); _ge_res["H2"].append((_e,rH2))

def _report(tag, A, B, rows):
    print(f"\n{tag}:  A={A}  B={B}")
    print(f"{'event':<14}{'N/arm':>6}{'collapse A':>12}{'collapse B':>12}{'A-B':>9}{'p':>8}")
    print("-"*62)
    ksig=0
    for _e,r in rows:
        if r is None: print(f"{_GE_LAB[_e]:<14}{'low n':>6}"); continue
        print(f"{_GE_LAB[_e]:<14}{r['N']:>6}{r['cA']:>12.3f}{r['cB']:>12.3f}{r['diff']:>+9.3f}{r['p']:>8.3f}{_star(r['p'])}")
        if r['p']<.05: ksig+=1
    print(f"  -> significant (p<.05) in {ksig}/{len(rows)} events")

_report("H1 Success vs Failure (strat by condition)","Success","Failure",_ge_res["H1"])
_report("H2 Visual vs Non-visual (balanced by outcome)","Visual(HUD+Full)","Non-visual(Base+Au)",_ge_res["H2"])
print("\nNotes:")
print(f" - Per-event test: {_GE_PERM} stratified group-label shuffles; ED at equal balanced n,")
print(f"   averaged over {_GE_R} subsample draws. Events = independent replications (k/3).")
print(" - Metric = collapse depth (Baseline-Action). H1 predicts larger collapse in one")
print("   outcome group; H2 predicts smaller collapse for visual warnings. Results in _ge_res.")
'''
ast.parse(code)

nb["cells"].insert(ins+1, {"cell_type":"markdown","metadata":{},"source":md})
nb["cells"].insert(ins+2, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted group ED phases cell after {ins}. Total cells: {len(nb['cells'])}.")
