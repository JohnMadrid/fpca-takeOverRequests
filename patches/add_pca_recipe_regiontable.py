import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA_temp_16.6.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already=any("REGION TABLE (PCA-RECIPE ED" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "cell present."

code = r'''# ============================================================
# REGION TABLE (PCA-RECIPE ED: per-driver StandardScaler, 8s window)
# ------------------------------------------------------------
# Recomputes ED(t) with the EXACT recipe of the "PCA -- CAR REFERENCE FRAME"
# cell (per-driver StandardScaler over 8s, then per-timepoint cross-driver
# center + per-channel z + participation-ratio ED), on the outliers-removed
# car-frame files. Then the region-label permutation table:
#   Contrast 1: Pre [-4,0) vs Post [0,4].
#   Contrast 2: Baseline / TOR / Action (boundary = mean RT_Steering per event);
#               pairwise diffs + omnibus. 5000 region-label shuffles.
# Self-contained.
# ============================================================
import numpy as np, os
import pandas as pd
import dask.dataframe as dd
from scipy.linalg import eigh as _eighR
from sklearn.preprocessing import StandardScaler

_RT_PERM=5000
_RT_RNG=np.random.default_rng(0)
_RT_VARS=["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_RT_M=len(_RT_VARS)
_RT_EVENTS=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_RT_LAB={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_RT_DIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/outliers_removed_5features/"

def _rt_edcurve(evt):
    """EXACT PCA-cell recipe: per-driver StandardScaler over 8s, then snapshot ED."""
    fp=_RT_DIR+f"car_reference_{evt}.csv"
    if not os.path.exists(fp): return None,None
    df=dd.read_csv(fp,assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()
    curves=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        raw=sub[_RT_VARS].values.astype(float)
        s0=np.linspace(-5,5,raw.shape[0]); raw=raw[(s0>=-4)&(s0<=4)]
        curves.append(StandardScaler().fit_transform(raw))
    arr=np.stack(curves,0); W,T,_=arr.shape
    secs=np.linspace(-5,5,points_per_window); secs=secs[(secs>=-4)&(secs<=4)]
    ed=np.zeros(T)
    for t in range(T):
        snap=arr[:,t,:].copy(); snap-=snap.mean(0)
        for m in range(_RT_M):
            sd=snap[:,m].std(ddof=1)
            if sd>1e-12: snap[:,m]/=sd
        eg=_eighR(snap.T@snap/(W-1),eigvals_only=True); p=eg/eg.sum()
        ed[t]=1.0/np.sum(p**2)
    return secs,ed

# mean RT_Steering per event (available only)
_rt=pd.read_csv(os.getcwd()+"/reaction_times.csv")
_rt["RT_Steering"]=pd.to_numeric(_rt["RT_Steering"],errors="coerce")
_rt=_rt[(_rt["Unavailable_Steering"].astype(str).str.lower()!="true") & _rt["RT_Steering"].notna()]
def _mrt(evt):
    sub=_rt[_rt["EventName"]==evt]; return float(sub["RT_Steering"].mean()) if len(sub) else np.nan

def _regions(secs,rt):
    return dict(Pre=(secs>=-4)&(secs<0),Post=(secs>=0)&(secs<=4),
               Baseline=(secs>=-4)&(secs<0),TOR=(secs>=0)&(secs<rt),Action=(secs>=rt)&(secs<=4))

def _perm2(ed,mA,mB,P=_RT_PERM,rng=None):
    rng=rng or _RT_RNG; idx=np.where(mA|mB)[0]; nA=int(mA.sum())
    obs=ed[mB].mean()-ed[mA].mean(); vals=ed[idx]; null=np.empty(P)
    for p in range(P):
        pm=rng.permutation(vals); null[p]=pm[nA:].mean()-pm[:nA].mean()
    return obs,float((np.sum(np.abs(null)>=abs(obs))+1)/(P+1))

def _perm3(ed,mB,mT,mA,P=_RT_PERM,rng=None):
    rng=rng or _RT_RNG; idx=np.where(mB|mT|mA)[0]; vals=ed[idx]
    nB=int(mB.sum()); nT=int(mT.sum())
    mb,mt,ma=ed[mB].mean(),ed[mT].mean(),ed[mA].mean()
    op={"Baseline-TOR":mb-mt,"TOR-Action":mt-ma,"Baseline-Action":mb-ma}; oo=np.var([mb,mt,ma])
    npp={k:np.empty(P) for k in op}; no=np.empty(P)
    for p in range(P):
        pm=rng.permutation(vals); b=pm[:nB].mean(); t=pm[nB:nB+nT].mean(); a=pm[nB+nT:].mean()
        npp["Baseline-TOR"][p]=b-t; npp["TOR-Action"][p]=t-a; npp["Baseline-Action"][p]=b-a; no[p]=np.var([b,t,a])
    pv={k:float((np.sum(np.abs(npp[k])>=abs(op[k]))+1)/(P+1)) for k in op}
    return op,pv,oo,float((np.sum(no>=oo)+1)/(P+1))

def _st(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""

_store={}
for _e in _RT_EVENTS:
    secs,ed=_rt_edcurve(_e)
    if secs is None: print(f"{_e}: not found"); continue
    _store[_e]=dict(secs=secs,ed=ed,rt=_mrt(_e))

print("="*78); print("CONTRAST 1: pre vs post (region-label shuffle, two-sided)"); print("="*78)
print(f"{'event':<14}{'mean Pre':>9}{'mean Post':>10}{'Post-Pre':>10}{'p':>8}"); print("-"*78)
_edavg=[]
for _e in _RT_EVENTS:
    if _e not in _store: continue
    d=_store[_e]; R=_regions(d['secs'],d['rt']); _edavg.append(d['ed'])
    o,pv=_perm2(d['ed'],R['Pre'],R['Post'])
    print(f"{_RT_LAB[_e]:<14}{d['ed'][R['Pre']].mean():>9.3f}{d['ed'][R['Post']].mean():>10.3f}{o:>+10.3f}{pv:>8.3f}{_st(pv)}")
if len(_edavg)>1:
    avg=np.mean(_edavg,0); secs=_store[_RT_EVENTS[0]]['secs']
    R=_regions(secs,np.nanmean([_store[e]['rt'] for e in _RT_EVENTS if e in _store]))
    o,pv=_perm2(avg,R['Pre'],R['Post'])
    print(f"{'AVERAGE':<14}{avg[R['Pre']].mean():>9.3f}{avg[R['Post']].mean():>10.3f}{o:>+10.3f}{pv:>8.3f}{_st(pv)}")

print("\n"+"="*78); print("CONTRAST 2: Baseline / TOR / Action (3-phase; pairwise + omnibus)")
print("  boundary = mean RT_Steering per event"); print("="*78)
print(f"{'event':<14}{'Base':>7}{'TOR':>7}{'Action':>7} | {'B-TOR(p)':>14}{'TOR-Act(p)':>15}{'B-Act(p)':>14}{'omni p':>9}")
print("-"*100)
for _e in _RT_EVENTS:
    if _e not in _store: continue
    d=_store[_e]; R=_regions(d['secs'],d['rt']); ed=d['ed']
    if R['TOR'].sum()<2 or R['Action'].sum()<2: print(f"{_RT_LAB[_e]:<14} TOR/Action too short"); continue
    mb,mt,ma=ed[R['Baseline']].mean(),ed[R['TOR']].mean(),ed[R['Action']].mean()
    op,pv,oo,po=_perm3(ed,R['Baseline'],R['TOR'],R['Action'])
    print(f"{_RT_LAB[_e]:<14}{mb:>7.2f}{mt:>7.2f}{ma:>7.2f} | "
          f"{op['Baseline-TOR']:>+7.2f}({pv['Baseline-TOR']:.3f}){_st(pv['Baseline-TOR']):<3}"
          f"{op['TOR-Action']:>+7.2f}({pv['TOR-Action']:.3f}){_st(pv['TOR-Action']):<3}"
          f"{op['Baseline-Action']:>+7.2f}({pv['Baseline-Action']:.3f}){_st(pv['Baseline-Action']):<3}"
          f"{po:>9.3f}{_st(po)}")
print("\nNotes:")
print(f" - PCA-recipe ED (per-driver StandardScaler, 8s window), outliers-removed car frame.")
print(f" - {_RT_PERM} region-label shuffles. Contrast1 & pairwise two-sided; omnibus one-tailed.")
print(" - * p<.05  ** p<.01  *** p<.001.")
'''
ast.parse(code)
nb["cells"].append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Appended PCA-recipe region table cell. Total: {len(nb['cells'])}.")
