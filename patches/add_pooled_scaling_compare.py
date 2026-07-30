import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already = any("SNAPSHOT PCA ED: SCALING-RECIPE COMPARISON" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source",""))
              for c in nb["cells"])
assert not already, "scaling-compare cell already present."

# insert after the outliers-removed PCA cell, before/near the Option cells
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "PCA CAR-FRAME (OUTLIERS REMOVED)" in "".join(c["source"]): ins=i
if ins is None: ins=len(nb["cells"])-1

md = r'''## Snapshot PCA ED: scaling-recipe comparison

The snapshot-PCA cells use per-driver StandardScaler (per-driver center AND
per-driver unit-variance scale). The per-driver scaling removes between-driver
amplitude, so ED measures waveform SHAPE, not behaviour-as-executed. This cell
compares three recipes so the effect of each preprocessing layer is visible:

1. **Per-driver center + per-driver scale** (current StandardScaler).
2. **Pooled** (cross-driver center + pooled per-channel scale, no per-driver C).
3. **Gold standard: per-driver center (C) + pooled per-channel scale** -- removes
   calibration/posture offsets via C, fixes between-channel scale via pooled
   scaling, KEEPS between-driver amplitude. Matches MFPCA preprocessing.

Pooled scaling fixes between-channel scale only; it does NOT remove per-driver
offsets (that needs centering). C + pooled is the recipe that does both.
'''

code = r'''# ============================================================
# SNAPSHOT PCA ED: SCALING-RECIPE COMPARISON (car frame)
# ------------------------------------------------------------
# ED(t) under three preprocessing recipes:
#   R1 perdriver_z : StandardScaler per driver (current). center+scale per driver.
#   R2 pooled      : cross-driver center at t + pooled per-channel scale. No C.
#   R3 C_pooled    : per-driver center (operation C) + pooled per-channel scale.
#                    GOLD: offsets removed (C), channel scale fixed (pooled),
#                    between-driver amplitude PRESERVED. Matches MFPCA.
# ED(t)=(sum lambda)^2/sum(lambda^2) per timepoint, [-4,4]s crop. No perms.
# ============================================================
import numpy as np, os
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scipy.linalg import eigh as _eighSC

_SC_VARS=["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_SC_M=len(_SC_VARS)
_SC_EVENTS=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_SC_LAB={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_SC_DIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/"

def _sc_load_raw(evt):
    """Per-driver RAW car-frame curves (no scaling). (W,501,M)."""
    fp=_SC_DIR+f"car_reference_{evt}.csv"
    if not os.path.exists(fp): return None,None
    df=dd.read_csv(fp,assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()
    curves=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        curves.append(sub.sort_values('time_from_event')[_SC_VARS].values.astype(float))
    arr=np.stack(curves,0); secs=np.linspace(-5,5,arr.shape[1]); crop=(secs>=-4)&(secs<=4)
    return arr[:,crop,:], secs[crop]

def _ed_of_snapshot(snap):
    W,M=snap.shape
    cov=snap.T@snap/(W-1)
    ev=np.maximum(_eighSC(cov,eigvals_only=True),0); s=ev.sum()
    return (s**2)/np.sum(ev**2) if s>0 else np.nan

def _ed_curve(arr, recipe):
    """arr: RAW (W,T,M). Returns ED(t)."""
    W,T,M=arr.shape
    A=arr.astype(float).copy()
    if recipe=="perdriver_z":
        # per-driver center + per-driver scale (StandardScaler equivalent)
        mu=A.mean(axis=1,keepdims=True); sd=A.std(axis=1,ddof=0,keepdims=True); sd[sd<1e-12]=1.0
        A=(A-mu)/sd
    elif recipe=="C_pooled":
        # per-driver center (C) only here; pooled scale applied per timepoint below
        A=A-A.mean(axis=1,keepdims=True)
    # pooled per-channel SD (over drivers x time) for pooled recipes
    if recipe in ("pooled","C_pooled"):
        flat=A.reshape(-1,M); psd=flat.std(0,ddof=1); psd[psd<1e-12]=1.0
    ed=np.zeros(T)
    for t in range(T):
        snap=A[:,t,:].copy()
        snap=snap-snap.mean(0)                    # cross-driver center at t (all recipes)
        if recipe=="perdriver_z":
            # already per-driver scaled; restandardise channel at t (matches current cell)
            for m in range(M):
                s=snap[:,m].std(ddof=1)
                if s>1e-12: snap[:,m]/=s
        else:
            snap=snap/psd                          # pooled per-channel scale
        ed[t]=_ed_of_snapshot(snap)
    return ed

_RECIPES=[("perdriver_z","per-driver z (current)","#888888"),
          ("pooled","pooled (no C)","#1f77b4"),
          ("C_pooled","C + pooled (gold)","#d62728")]

print("SNAPSHOT PCA ED: scaling-recipe comparison (car frame)")
print(f"{'event':<14}{'recipe':<22}{'pre':>7}{'post':>7}{'drop':>8}")
print("-"*60)
_sc_store={}
for _e in _SC_EVENTS:
    arr,secs=_sc_load_raw(_e)
    if arr is None: print(f"  {_e}: not found"); continue
    pre=secs<0; post=secs>=0; _sc_store[_e]=dict(secs=secs,curves={})
    for key,lab,col in _RECIPES:
        ed=_ed_curve(arr,key); _sc_store[_e]["curves"][key]=ed
        print(f"{_SC_LAB[_e]:<14}{lab:<22}{ed[pre].mean():>7.2f}{ed[post].mean():>7.2f}{ed[post].mean()-ed[pre].mean():>+8.2f}")
    print("-"*60)

# overlay plot per event
for _e,d in _sc_store.items():
    fig,ax=plt.subplots(figsize=(8,4.6))
    for key,lab,col in _RECIPES:
        ed=d["curves"][key]; ax.plot(d["secs"],ed,color=col,lw=1.6,label=lab)
        ax.axhline(ed[d["secs"]<0].mean(),color=col,ls=":",lw=0.8,alpha=0.6)
    ax.axvline(0,color="k",lw=0.8,ls="--")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Effective dimensionality")
    ax.set_title(f"{_SC_LAB[_e]} -- ED(t) by scaling recipe (car frame)",fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

print("\\nReading:")
print(" - per-driver z removes between-driver amplitude (ED = waveform shape only).")
print(" - C+pooled keeps amplitude, removes per-driver offsets, fixes channel scale.")
print(" - If the post-onset drop shrinks/reverses under C+pooled, the current drop")
print("   was partly an amplitude-blinding artifact of per-driver scaling.")
'''
ast.parse(code)

nb["cells"].insert(ins+1, {"cell_type":"markdown","metadata":{},"source":md})
nb["cells"].insert(ins+2, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted scaling-recipe comparison after {ins}. Total cells: {len(nb['cells'])}.")
