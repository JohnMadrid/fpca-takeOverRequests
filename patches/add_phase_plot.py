import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA_temp_16.6.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already=any("PHASE-MEAN PLOT (Baseline/TOR/Action)" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "cell present."

code = r'''# ============================================================
# PHASE-MEAN PLOT (Baseline/TOR/Action) -- 3 events + average, one figure
# ------------------------------------------------------------
# PCA-recipe ED (per-driver StandardScaler, 8s window, outliers-removed car frame).
# x = 3 phases; one coloured line per event; thick black line = average.
# Stars on Base->TOR and TOR->Action segments = pairwise region-shuffle p.
# Reuses _store / helpers from the REGION TABLE cell if present; else recomputes.
# ============================================================
import numpy as np, os
import matplotlib.pyplot as plt

# --- ensure we have ED curves + rt per event (reuse _store if the table cell ran) ---
if "_store" not in globals() or not _store:
    import pandas as pd, dask.dataframe as dd
    from scipy.linalg import eigh as _eighP
    from sklearn.preprocessing import StandardScaler
    _PVARS=["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
    _PDIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/outliers_removed_5features/"
    _rt=pd.read_csv(os.getcwd()+"/reaction_times.csv")
    _rt["RT_Steering"]=pd.to_numeric(_rt["RT_Steering"],errors="coerce")
    _rt=_rt[(_rt["Unavailable_Steering"].astype(str).str.lower()!="true") & _rt["RT_Steering"].notna()]
    def _mrt(e):
        s=_rt[_rt["EventName"]==e]; return float(s["RT_Steering"].mean()) if len(s) else np.nan
    def _edc(e):
        df=dd.read_csv(_PDIR+f"car_reference_{e}.csv",assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()
        cs=[]
        for _,sub in df.groupby('uid'):
            if len(sub)!=points_per_window: continue
            raw=sub[_PVARS].values.astype(float); s0=np.linspace(-5,5,raw.shape[0]); raw=raw[(s0>=-4)&(s0<=4)]
            cs.append(StandardScaler().fit_transform(raw))
        arr=np.stack(cs,0); W,T,_=arr.shape; secs=np.linspace(-5,5,points_per_window); secs=secs[(secs>=-4)&(secs<=4)]
        ed=np.zeros(T)
        for t in range(T):
            sn=arr[:,t,:].copy(); sn-=sn.mean(0)
            for m in range(len(_PVARS)):
                sd=sn[:,m].std(ddof=1)
                if sd>1e-12: sn[:,m]/=sd
            eg=_eighP(sn.T@sn/(W-1),eigvals_only=True); p=eg/eg.sum(); ed[t]=1.0/np.sum(p**2)
        return secs,ed
    _store={}
    for e in ["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]:
        secs,ed=_edc(e); _store[e]=dict(secs=secs,ed=ed,rt=_mrt(e))

_PLAB={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_PCOL={"StagEventNew":"steelblue","FallingRocksEventNew":"#2ca02c","MotorcyclistEvent":"#d62728"}
_PEV=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_PERM=5000; _RNG=np.random.default_rng(0)

def _regions(secs,rt):
    return dict(Baseline=(secs>=-4)&(secs<0),TOR=(secs>=0)&(secs<rt),Action=(secs>=rt)&(secs<=4))
def _perm_pair(ed,mA,mB):
    idx=np.where(mA|mB)[0]; nA=int(mA.sum()); obs=ed[mA].mean()-ed[mB].mean(); vals=ed[idx]; null=np.empty(_PERM)
    for p in range(_PERM):
        pm=_RNG.permutation(vals); null[p]=pm[:nA].mean()-pm[nA:].mean()
    return obs,float((np.sum(np.abs(null)>=abs(obs))+1)/(_PERM+1))
def _st(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"

xpos=[0,1,2]; xlab=["Baseline","TOR","Action"]
fig,ax=plt.subplots(figsize=(8.5,6))
_phase_means={}
for e in _PEV:
    if e not in _store: continue
    d=_store[e]; R=_regions(d['secs'],d['rt']); ed=d['ed']
    m=[ed[R['Baseline']].mean(),ed[R['TOR']].mean(),ed[R['Action']].mean()]
    _phase_means[e]=m
    ax.plot(xpos,m,'o-',color=_PCOL[e],lw=2,ms=8,label=_PLAB[e])
    # pairwise stars on segments
    _,p_bt=_perm_pair(ed,R['Baseline'],R['TOR'])
    _,p_ta=_perm_pair(ed,R['TOR'],R['Action'])
    ax.annotate(_st(p_bt),((xpos[0]+xpos[1])/2,(m[0]+m[1])/2),fontsize=10,color=_PCOL[e],
                ha="center",va="bottom",fontweight="bold")
    ax.annotate(_st(p_ta),((xpos[1]+xpos[2])/2,(m[1]+m[2])/2),fontsize=10,color=_PCOL[e],
                ha="center",va="bottom",fontweight="bold")

# average across events (thick black)
if _phase_means:
    avg=np.mean([_phase_means[e] for e in _phase_means],0)
    ax.plot(xpos,avg,'o-',color="k",lw=3.2,ms=9,label="Average",zorder=5)
    # average pairwise stars from averaged ED curve
    secs=_store[_PEV[0]]['secs']; rt_avg=np.nanmean([_store[e]['rt'] for e in _store])
    edavg=np.mean([_store[e]['ed'] for e in _store],0); Ra=_regions(secs,rt_avg)
    _,pbt=_perm_pair(edavg,Ra['Baseline'],Ra['TOR']); _,pta=_perm_pair(edavg,Ra['TOR'],Ra['Action'])
    ax.annotate(_st(pbt),((xpos[0]+xpos[1])/2,(avg[0]+avg[1])/2),fontsize=11,color="k",ha="center",va="top",fontweight="bold")
    ax.annotate(_st(pta),((xpos[1]+xpos[2])/2,(avg[1]+avg[2])/2),fontsize=11,color="k",ha="center",va="top",fontweight="bold")

ax.set_xticks(xpos); ax.set_xticklabels(xlab,fontsize=11)
ax.set_ylabel("Effective dimensionality (PCA-recipe ED)",fontsize=11)
ax.set_title("ED by phase: Baseline / Take-over request / Action\n(stars = pairwise region-shuffle p)",fontsize=12,fontweight="bold")
ax.grid(True,alpha=0.3); ax.legend(fontsize=10)
plt.tight_layout(); plt.show()
print("Stars: *** p<.001  ** p<.01  * p<.05  ns. Boundary TOR/Action = mean RT_Steering per event.")
'''
ast.parse(code)
nb["cells"].append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Appended phase-mean plot cell. Total: {len(nb['cells'])}.")
