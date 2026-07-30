import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already = any("OPTION 1: TASK-ALIGNED vs ORTHOGONAL" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source",""))
              for c in nb["cells"])
assert not already, "UCM/ED-shape cells already present."

# insert after the outliers-removed PCA cell (fallback: end)
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "PCA CAR-FRAME (OUTLIERS REMOVED)" in "".join(c["source"]): ins=i
if ins is None: ins=len(nb["cells"])-1

# ---------------- Option 1: task-aligned vs orthogonal variance ----------------
md1 = r'''## Option 1: task-aligned vs orthogonal variance (UCM-style)

Decompose post-onset between-driver variance into a **task axis** and the rest, to
ask WHY snapshot ED drops:

- **collapse onto the axis** (aligned variance fraction rises) -> drivers converge
  on a shared steering-gaze solution (coordination), or
- **off-manifold cleanup** (orthogonal variance shrinks, aligned flat) -> the
  classic uncontrolled-manifold signature: task-irrelevant exploration is
  suppressed while the task-relevant subspace stays free.

Task axis: (b) leading eigenvector of the per-timepoint cross-channel covariance
(method-internal "dominant coordination mode"); (a) steering-regression split as
a cross-check (fraction of gaze/head variance predictable from SteeringInput).
Car-frame channels.
'''

code1 = r'''# ============================================================
# OPTION 1: TASK-ALIGNED vs ORTHOGONAL variance over time (car frame)
# ------------------------------------------------------------
# Per timepoint, (W drivers x 5 channels) covariance.
#  (b) task axis = leading eigenvector; aligned_frac = lambda_1 / trace.
#  (a) steering split = R^2 of gaze/head regressed on SteeringInput
#      (fraction of non-steering variance explained by steering).
# Rising aligned_frac post-onset = collapse-onto-axis; falling orthogonal with
# flat aligned = off-manifold (UCM) cleanup.
# Self-contained; reads car_reference CSVs.
# ============================================================
import numpy as np, os
import dask.dataframe as dd
import matplotlib.pyplot as plt
from numpy.linalg import eigh as _eighU
from sklearn.preprocessing import StandardScaler

_U_VARS = ["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_U_STEER = _U_VARS.index("SteeringInput")
_U_GAZE  = [i for i in range(len(_U_VARS)) if i!=_U_STEER]
_U_EVENTS=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_U_LAB={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_U_COL={"StagEventNew":"steelblue","FallingRocksEventNew":"#2ca02c","MotorcyclistEvent":"#d62728"}
_U_DIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/"

def _u_load(evt):
    fp=_U_DIR+f"car_reference_{evt}.csv"
    if not os.path.exists(fp): return None,None
    df=dd.read_csv(fp,assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()
    curves=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        curves.append(StandardScaler().fit_transform(sub.sort_values('time_from_event')[_U_VARS].values))
    arr=np.stack(curves,0)                      # (W,501,M) per-driver standardised
    secs=np.linspace(-5,5,arr.shape[1]); crop=(secs>=-4)&(secs<=4)
    return arr[:,crop,:], secs[crop]

def _aligned_orth(snap):
    """snap: (W,M) at one timepoint, cross-driver mean removed + unit-var per chan.
    Returns (aligned_frac_b, ed, steer_R2)."""
    X=snap-snap.mean(0)
    for m in range(X.shape[1]):
        sd=X[:,m].std(ddof=1)
        if sd>1e-12: X[:,m]/=sd
    cov=np.cov(X.T); ev=np.sort(np.maximum(_eighU(cov)[0],0))[::-1]
    tr=ev.sum()
    aligned_b=float(ev[0]/tr) if tr>0 else np.nan          # leading-mode fraction
    ed=float((tr**2)/np.sum(ev**2)) if tr>0 else np.nan
    # steering split: regress each gaze/head channel on steering, mean R^2
    s=X[:,_U_STEER]; ss=s@s
    if ss>1e-12:
        r2s=[]
        for g in _U_GAZE:
            b=(s@X[:,g])/ss; pred=b*s
            sst=np.sum((X[:,g]-X[:,g].mean())**2)
            r2s.append(1.0-np.sum((X[:,g]-pred)**2)/sst if sst>1e-12 else np.nan)
        steer_r2=float(np.nanmean(r2s))
    else:
        steer_r2=np.nan
    return aligned_b, ed, steer_r2

print("OPTION 1: task-aligned vs orthogonal variance (car frame)")
_u_store={}
for _e in _U_EVENTS:
    arr,secs=_u_load(_e)
    if arr is None: print(f"  {_e}: not found"); continue
    T=arr.shape[1]; al=np.zeros(T); ed=np.zeros(T); r2=np.zeros(T)
    for t in range(T):
        al[t],ed[t],r2[t]=_aligned_orth(arr[:,t,:])
    _u_store[_e]=dict(secs=secs,aligned=al,orth=1-al,ed=ed,steer_r2=r2)
    pre=secs<0; post=secs>=0
    print(f"  {_U_LAB[_e]:<14} aligned_frac pre={al[pre].mean():.3f} post={al[post].mean():.3f} "
          f"| orth pre={(1-al)[pre].mean():.3f} post={(1-al)[post].mean():.3f} "
          f"| steerR2 pre={np.nanmean(r2[pre]):.3f} post={np.nanmean(r2[post]):.3f}")

# plots: aligned vs orthogonal fraction + ED + steering R2
for _e,d in _u_store.items():
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    fig.suptitle(f"{_U_LAB[_e]} -- task-aligned vs orthogonal (car frame)",fontweight="bold")
    ax=axes[0]
    ax.plot(d["secs"],d["aligned"],color=_U_COL[_e],lw=1.6,label="aligned (leading mode)")
    ax.plot(d["secs"],d["orth"],color="gray",lw=1.4,ls="--",label="orthogonal (rest)")
    ax.axvline(0,color="k",lw=0.8,ls="--"); ax.set_ylim(0,1); ax.set_xlabel("Time (s)")
    ax.set_ylabel("variance fraction"); ax.set_title("(b) covariance-axis split"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax=axes[1]
    ax.plot(d["secs"],d["ed"],color=_U_COL[_e],lw=1.6)
    pre=d["secs"]<0; ax.axhline(d["ed"][pre].mean(),color="tomato",ls="--",lw=1)
    ax.axvline(0,color="k",lw=0.8,ls="--"); ax.set_xlabel("Time (s)"); ax.set_ylabel("ED"); ax.set_title("snapshot ED(t)"); ax.grid(alpha=0.3)
    ax=axes[2]
    ax.plot(d["secs"],d["steer_r2"],color=_U_COL[_e],lw=1.6)
    ax.axvline(0,color="k",lw=0.8,ls="--"); ax.set_ylim(0,1); ax.set_xlabel("Time (s)")
    ax.set_ylabel("mean R2"); ax.set_title("(a) gaze/head variance from steering"); ax.grid(alpha=0.3)
    plt.tight_layout(rect=[0,0,1,0.94]); plt.show()

print("\\nReading: aligned_frac UP post-onset -> collapse onto a shared axis (coordination).")
print("         orthogonal DOWN with aligned flat -> UCM off-manifold cleanup.")
print("         steering R2 UP -> the shared axis is steering-coupled gaze/head.")
'''
ast.parse(code1)

# ---------------- Option 3: ED(t) curve-shape metrics + RT correlation ----------------
md3 = r'''## Option 3: ED(t) dynamics -- shape metrics and reaction-time link

Treat the snapshot ED(t) curve as the object. Per event: trough depth, time-to-
trough (latency), collapse rate, recovery rate, AUC below pre-mean, FWHM of the
dip. Then correlate ED-trough latency against mean `RT_Steering` (the notebook's
standard RT measure, from reaction_times.csv). If the dimensionality trough lags
onset by ~the steering reaction time, the collapse tracks the motor response.
'''

code3 = r'''# ============================================================
# OPTION 3: ED(t) SHAPE METRICS + reaction-time link (car frame)
# ------------------------------------------------------------
# Recomputes snapshot ED(t) (self-contained), extracts curve-shape metrics, and
# correlates ED-trough latency with mean RT_Steering (reaction_times.csv).
# ============================================================
import numpy as np, os
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scipy.linalg import eigh as _eighS
from sklearn.preprocessing import StandardScaler

_S_VARS=["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_S_EVENTS=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_S_LAB={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_S_COL={"StagEventNew":"steelblue","FallingRocksEventNew":"#2ca02c","MotorcyclistEvent":"#d62728"}
_S_DIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/"

def _s_edcurve(evt):
    fp=_S_DIR+f"car_reference_{evt}.csv"
    if not os.path.exists(fp): return None,None
    df=dd.read_csv(fp,assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()
    curves=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        curves.append(StandardScaler().fit_transform(sub.sort_values('time_from_event')[_S_VARS].values))
    arr=np.transpose(np.stack(curves,0),(0,2,1))   # (W,T,M)->? keep (W,501,M)
    arr=np.stack(curves,0)
    secs=np.linspace(-5,5,arr.shape[1]); crop=(secs>=-4)&(secs<=4)
    arr=arr[:,crop,:]; secs=secs[crop]; W,T,M=arr.shape
    ed=np.zeros(T)
    for t in range(T):
        snap=arr[:,t,:].copy(); snap-=snap.mean(0)
        for m in range(M):
            sd=snap[:,m].std(ddof=1)
            if sd>1e-12: snap[:,m]/=sd
        ev=_eighS(snap.T@snap/(W-1),eigvals_only=True); p=ev/ev.sum()
        ed[t]=1.0/np.sum(p**2)
    return secs,ed

def _shape_metrics(secs,ed):
    dt=secs[1]-secs[0]; pre=secs<0; post=secs>=0
    pre_mean=ed[pre].mean()
    post_ed=ed[post]; post_secs=secs[post]
    ti=int(np.argmin(post_ed)); trough_t=post_secs[ti]; trough_v=post_ed[ti]
    depth=trough_v-pre_mean
    # collapse rate: onset(0) to trough
    onset_v=ed[np.argmin(np.abs(secs))]
    collapse_rate=(trough_v-onset_v)/max(trough_t,1e-9)
    # recovery: trough to end
    end_v=post_ed[-1]; rec_time=post_secs[-1]-trough_t
    recovery_rate=(end_v-trough_v)/max(rec_time,1e-9)
    # time back to within 5% of pre-mean after trough
    after=post_secs>trough_t
    back=np.where(after & (post_ed>=pre_mean-0.05*abs(pre_mean)))[0]
    t_recover=post_secs[back[0]]-0.0 if len(back) else np.nan
    # AUC below pre-mean (post)
    below=np.clip(pre_mean-post_ed,0,None); auc=below.sum()*dt
    # FWHM of the dip at half depth
    half=pre_mean+0.5*depth   # depth negative -> half between pre and trough
    inside=post_ed<=half
    if inside.any():
        idx=np.where(inside)[0]; fwhm=(post_secs[idx[-1]]-post_secs[idx[0]])
    else:
        fwhm=np.nan
    return dict(pre_mean=pre_mean,trough_t=trough_t,trough_v=trough_v,depth=depth,
                collapse_rate=collapse_rate,recovery_rate=recovery_rate,
                t_recover=t_recover,auc=auc,fwhm=fwhm)

# --- RT lookup (RT_Steering, available only) ---
_rt=pd.read_csv(os.getcwd()+"/reaction_times.csv")
_rt["RT_Steering"]=pd.to_numeric(_rt["RT_Steering"],errors="coerce")
_rt_avail=_rt[(_rt["Unavailable_Steering"].astype(str).str.lower()!="true") & _rt["RT_Steering"].notna()]
def _mean_rt(evt):
    sub=_rt_avail[_rt_avail["EventName"]==evt]
    return float(sub["RT_Steering"].mean()), float(sub["RT_Steering"].median()), int(len(sub))

print("OPTION 3: ED(t) shape metrics (car frame)")
print(f"{'event':<14}{'depth':>7}{'t_trough':>9}{'collapse':>9}{'recov':>8}{'t_rec':>7}{'AUC':>7}{'FWHM':>7}{'meanRT':>8}")
print("-"*84)
_s_store={}
for _e in _S_EVENTS:
    secs,ed=_s_edcurve(_e)
    if secs is None: print(f"  {_e}: not found"); continue
    m=_shape_metrics(secs,ed); rt_mean,rt_med,rt_n=_mean_rt(_e)
    m.update(meanRT=rt_mean,medRT=rt_med,nRT=rt_n,secs=secs,ed=ed)
    _s_store[_e]=m
    print(f"{_S_LAB[_e]:<14}{m['depth']:>+7.2f}{m['trough_t']:>9.2f}{m['collapse_rate']:>9.2f}"
          f"{m['recovery_rate']:>8.2f}{m['t_recover']:>7.2f}{m['auc']:>7.2f}{m['fwhm']:>7.2f}{rt_mean:>8.2f}")

# plot ED(t) with trough + RT marker
fig,axes=plt.subplots(1,len(_s_store),figsize=(5*len(_s_store),4),squeeze=False)
for ci,(_e,m) in enumerate(_s_store.items()):
    ax=axes[0,ci]
    ax.plot(m["secs"],m["ed"],color=_S_COL[_e],lw=1.6)
    ax.axhline(m["pre_mean"],color="tomato",ls="--",lw=1,label=f"pre mean {m['pre_mean']:.2f}")
    ax.axvline(0,color="k",lw=0.8,ls="--")
    ax.plot(m["trough_t"],m["trough_v"],'v',color="black",ms=9,label=f"trough {m['trough_t']:.2f}s")
    ax.axvline(m["meanRT"],color="purple",lw=1.4,ls=":",label=f"mean RT {m['meanRT']:.2f}s")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("ED"); ax.set_title(_S_LAB[_e]); ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# correlation: trough latency vs mean RT across events (n=3 -> descriptive)
_tt=np.array([m["trough_t"] for m in _s_store.values()])
_rt_arr=np.array([m["meanRT"] for m in _s_store.values()])
if len(_tt)>=2:
    r=np.corrcoef(_tt,_rt_arr)[0,1]
    print(f"\\nED-trough latency vs mean RT_Steering across {len(_tt)} events: r={r:.2f}")
    print("(n=3 events -> descriptive only; per-driver version needed for inference.)")
    for _e,m in _s_store.items():
        print(f"  {_S_LAB[_e]:<14} trough={m['trough_t']:.2f}s  meanRT={m['meanRT']:.2f}s  diff={m['trough_t']-m['meanRT']:+.2f}s")
'''
ast.parse(code3)

cells=[
    {"cell_type":"markdown","metadata":{},"source":md1},
    {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code1},
    {"cell_type":"markdown","metadata":{},"source":md3},
    {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code3},
]
for off,cell in enumerate(cells,1):
    nb["cells"].insert(ins+off,cell)
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted Option 1 + Option 3 cells after {ins}. Total cells: {len(nb['cells'])}.")
