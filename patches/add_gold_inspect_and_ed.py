import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already = any("GOLD-STANDARD CHANNEL INSPECTION" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source",""))
              for c in nb["cells"])
assert not already, "gold inspect/ed cells already present."

ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "SCALING-RECIPE COMPARISON" in "".join(c["source"]): ins=i
if ins is None: ins=len(nb["cells"])-1

# ---------------- shared gold preprocessing note ----------------
md1 = r'''## Gold-standard channel inspection (data entering the PCA)

Each car-frame channel for ALL drivers as it enters the snapshot PCA under the
gold recipe (per-driver center = operation C, then pooled per-channel scaling).
All driver traces overlaid (thin) with the cross-driver mean (thick). Inspect for
outliers, flat channels, onset structure, and between-driver spread before
trusting the ED.
'''

code1 = r'''# ============================================================
# GOLD-STANDARD CHANNEL INSPECTION (car frame): data entering the PCA
# ------------------------------------------------------------
# Gold recipe: per-driver center (operation C) + pooled per-channel scale.
# Plots each of the 5 channels, all drivers overlaid + cross-driver mean.
# ============================================================
import numpy as np, os
import dask.dataframe as dd
import matplotlib.pyplot as plt

_GVARS=["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_GLAB=["Head.x (car)","Head.y","Eye.x (car)","Eye.y","Steering"]
_GM=len(_GVARS)
_GEVENTS=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_GENAME={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_GDIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/"

def _g_load_raw(evt):
    fp=_GDIR+f"car_reference_{evt}.csv"
    if not os.path.exists(fp): return None,None
    df=dd.read_csv(fp,assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()
    curves=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        curves.append(sub.sort_values('time_from_event')[_GVARS].values.astype(float))
    arr=np.stack(curves,0); secs=np.linspace(-5,5,arr.shape[1]); crop=(secs>=-4)&(secs<=4)
    return arr[:,crop,:], secs[crop]

def _gold_transform(arr):
    """Per-driver center (C) + pooled per-channel scale. Returns transformed (W,T,M)."""
    A=arr.astype(float)-arr.astype(float).mean(axis=1,keepdims=True)   # C
    flat=A.reshape(-1,A.shape[2]); psd=flat.std(0,ddof=1); psd[psd<1e-12]=1.0
    return A/psd

for _e in _GEVENTS:
    arr,secs=_g_load_raw(_e)
    if arr is None: print(f"  {_e}: not found"); continue
    G=_gold_transform(arr); W=G.shape[0]
    fig,axes=plt.subplots(1,_GM,figsize=(3.4*_GM,3.8),squeeze=False)
    fig.suptitle(f"{_GENAME[_e]} -- channels entering PCA (gold: C + pooled scale, W={W})",
                 fontsize=12,fontweight="bold")
    for ci in range(_GM):
        ax=axes[0,ci]
        ax.plot(secs,G[:,:,ci].T,color="#4477aa",lw=0.4,alpha=0.18)
        ax.plot(secs,G[:,:,ci].mean(0),color="#cc3311",lw=2.0)
        ax.axvline(0,color="k",lw=0.8,ls="--"); ax.axhline(0,color="gray",lw=0.5)
        ax.set_title(_GLAB[ci],fontsize=9); ax.set_xlabel("Time (s)")
        if ci==0: ax.set_ylabel("gold-scaled value")
        ax.grid(alpha=0.25)
    plt.tight_layout(rect=[0,0,1,0.92]); plt.show()
    # quick per-channel between-driver spread pre vs post
    pre=secs<0; post=secs>=0
    print(f"{_GENAME[_e]}: per-channel cross-driver SD (pre | post)")
    for ci in range(_GM):
        sd_pre=G[:,pre,ci].std(0).mean(); sd_post=G[:,post,ci].std(0).mean()
        print(f"   {_GLAB[ci]:<14} {sd_pre:.3f} | {sd_post:.3f}")
    print()
'''
ast.parse(code1)

# ---------------- ED pre/post under gold ----------------
md2 = r'''## ED pre vs post (gold standard: C + pooled scale)

Snapshot PCA ED(t) under the gold recipe, with the post-onset region shaded where
ED falls below the pre-onset mean. Per event + average across events. This is the
amplitude-preserving, offset-removed, channel-scale-fair version of the headline
ED result.
'''

code2 = r'''# ============================================================
# ED PRE vs POST -- GOLD STANDARD (C + pooled per-channel scale, car frame)
# ============================================================
import numpy as np, os
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scipy.linalg import eigh as _eighG

_EVARS=["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_EEVENTS=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_EENAME={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_ECOL={"StagEventNew":"steelblue","FallingRocksEventNew":"#2ca02c","MotorcyclistEvent":"#d62728"}
_EDIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/"

def _e_load_raw(evt):
    fp=_EDIR+f"car_reference_{evt}.csv"
    if not os.path.exists(fp): return None,None
    df=dd.read_csv(fp,assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()
    curves=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        curves.append(sub.sort_values('time_from_event')[_EVARS].values.astype(float))
    arr=np.stack(curves,0); secs=np.linspace(-5,5,arr.shape[1]); crop=(secs>=-4)&(secs<=4)
    return arr[:,crop,:], secs[crop]

def _ed_curve_gold(arr):
    A=arr.astype(float)-arr.astype(float).mean(axis=1,keepdims=True)        # C
    flat=A.reshape(-1,A.shape[2]); psd=flat.std(0,ddof=1); psd[psd<1e-12]=1.0
    W,T,M=A.shape; ed=np.zeros(T)
    for t in range(T):
        snap=A[:,t,:].copy(); snap=snap-snap.mean(0); snap=snap/psd
        cov=snap.T@snap/(W-1); ev=np.maximum(_eighG(cov,eigvals_only=True),0); s=ev.sum()
        ed[t]=(s**2)/np.sum(ev**2) if s>0 else np.nan
    return ed

def _plot_ed(ax,secs,ed,color,title):
    pre=secs<0; post=secs>=0; pm=ed[pre].mean()
    below=post&(ed<pm)
    ax.plot(secs,ed,color=color,lw=1.6,label="Observed ED")
    ax.fill_between(secs,ed,pm,where=below,alpha=0.25,color="tomato",label=f"Below pre mean ({pm:.2f})")
    ax.axhline(pm,color="tomato",ls="--",lw=1)
    ax.axvline(0,color="k",lw=0.8,ls="--")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Effective dimensionality"); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

_e_curves=[]; _e_secs=None
print("ED pre vs post (gold: C + pooled scale)")
print(f"{'event':<14}{'pre':>7}{'post':>7}{'drop':>8}{'dur<pre':>9}")
print("-"*46)
for _e in _EEVENTS:
    arr,secs=_e_load_raw(_e)
    if arr is None: print(f"  {_e}: not found"); continue
    ed=_ed_curve_gold(arr); _e_curves.append(ed); _e_secs=secs
    pre=secs<0; post=secs>=0; pm=ed[pre].mean()
    dur=(post&(ed<pm)).sum()*(secs[1]-secs[0])
    print(f"{_EENAME[_e]:<14}{ed[pre].mean():>7.2f}{ed[post].mean():>7.2f}{ed[post].mean()-ed[pre].mean():>+8.2f}{dur:>9.2f}")
    fig,ax=plt.subplots(figsize=(8,4.6))
    _plot_ed(ax,secs,ed,_ECOL[_e],f"{_EENAME[_e]} -- ED gold (C + pooled)")
    plt.tight_layout(); plt.show()

# average across events
if _e_curves:
    ed_avg=np.mean(_e_curves,axis=0)
    pre=_e_secs<0; post=_e_secs>=0; pm=ed_avg[pre].mean()
    dur=(post&(ed_avg<pm)).sum()*(_e_secs[1]-_e_secs[0])
    print(f"{'AVERAGE':<14}{ed_avg[pre].mean():>7.2f}{ed_avg[post].mean():>7.2f}{ed_avg[post].mean()-ed_avg[pre].mean():>+8.2f}{dur:>9.2f}")
    fig,ax=plt.subplots(figsize=(8,4.6))
    _plot_ed(ax,_e_secs,ed_avg,"#444444","Average across events -- ED gold (C + pooled)")
    plt.tight_layout(); plt.show()
'''
ast.parse(code2)

cells=[
    {"cell_type":"markdown","metadata":{},"source":md1},
    {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code1},
    {"cell_type":"markdown","metadata":{},"source":md2},
    {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code2},
]
for off,cell in enumerate(cells,1):
    nb["cells"].insert(ins+off,cell)
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted gold inspection + ED cells after {ins}. Total cells: {len(nb['cells'])}.")
