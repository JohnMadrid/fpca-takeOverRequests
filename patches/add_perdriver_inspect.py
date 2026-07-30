import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already = any("PER-DRIVER-Z CHANNEL INSPECTION" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source",""))
              for c in nb["cells"])
assert not already, "per-driver inspect cell already present."

# insert right after the GOLD-STANDARD CHANNEL INSPECTION code cell
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "GOLD-STANDARD CHANNEL INSPECTION" in "".join(c["source"]): ins=i
assert ins is not None, "gold inspection cell not found"

md = r'''## Per-driver-z channel inspection (data entering the PCA, current recipe)

Same channel inspection but under the CURRENT recipe (per-driver StandardScaler:
per-driver center AND per-driver unit-variance scale, over the 8s window). Compare
against the gold inspection above: per-driver z forces every driver's channel to
unit variance, flattening between-driver amplitude differences (every trace fills
the same vertical range), whereas gold (C + pooled) preserves them.
'''

code = r'''# ============================================================
# PER-DRIVER-Z CHANNEL INSPECTION (car frame): data entering the PCA
# ------------------------------------------------------------
# Current recipe: per-driver StandardScaler (center + scale per driver) over the
# 8s window. All drivers overlaid + cross-driver mean. Compare to gold inspection.
# ============================================================
import numpy as np, os
import dask.dataframe as dd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

_PVARS=["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_PLAB=["Head.x (car)","Head.y","Eye.x (car)","Eye.y","Steering"]
_PM=len(_PVARS)
_PEVENTS=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_PENAME={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_PDIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/"

def _p_load_perdriverz(evt):
    """Per-driver z over the 8s window (crop first, then StandardScaler). (W,T,M), secs."""
    fp=_PDIR+f"car_reference_{evt}.csv"
    if not os.path.exists(fp): return None,None
    df=dd.read_csv(fp,assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()
    curves=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        _raw=sub.sort_values('time_from_event')[_PVARS].values.astype(float)
        _s0=np.linspace(-5,5,_raw.shape[0]); _raw=_raw[(_s0>=-4)&(_s0<=4)]
        curves.append(StandardScaler().fit_transform(_raw))     # per-driver z over 8s
    arr=np.stack(curves,0)
    secs=np.linspace(-5,5,points_per_window); secs=secs[(secs>=-4)&(secs<=4)]
    return arr, secs

for _e in _PEVENTS:
    arr,secs=_p_load_perdriverz(_e)
    if arr is None: print(f"  {_e}: not found"); continue
    W=arr.shape[0]
    fig,axes=plt.subplots(1,_PM,figsize=(3.4*_PM,3.8),squeeze=False)
    fig.suptitle(f"{_PENAME[_e]} -- channels entering PCA (per-driver z over 8s, W={W})",
                 fontsize=12,fontweight="bold")
    for ci in range(_PM):
        ax=axes[0,ci]
        ax.plot(secs,arr[:,:,ci].T,color="#999933",lw=0.4,alpha=0.18)
        ax.plot(secs,arr[:,:,ci].mean(0),color="#cc3311",lw=2.0)
        ax.axvline(0,color="k",lw=0.8,ls="--"); ax.axhline(0,color="gray",lw=0.5)
        ax.set_title(_PLAB[ci],fontsize=9); ax.set_xlabel("Time (s)")
        if ci==0: ax.set_ylabel("per-driver z value")
        ax.grid(alpha=0.25)
    plt.tight_layout(rect=[0,0,1,0.92]); plt.show()
    pre=secs<0; post=secs>=0
    print(f"{_PENAME[_e]}: per-channel cross-driver SD (pre | post)  [per-driver z]")
    for ci in range(_PM):
        print(f"   {_PLAB[ci]:<14} {arr[:,pre,ci].std(0).mean():.3f} | {arr[:,post,ci].std(0).mean():.3f}")
    print()
print("Note: per-driver z flattens between-driver amplitude (every driver unit-variance),")
print("so cross-driver SD here reflects only SHAPE/phase differences, not magnitude.")
'''
ast.parse(code)

nb["cells"].insert(ins+1, {"cell_type":"markdown","metadata":{},"source":md})
nb["cells"].insert(ins+2, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted per-driver-z inspection after {ins}. Total cells: {len(nb['cells'])}.")
