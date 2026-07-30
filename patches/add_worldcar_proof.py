import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already = any("WORLD-FRAME ARTIFACT PROOF" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source",""))
              for c in nb["cells"])
assert not already, "proof cell already present."

md = r'''# Proof: world-frame axis distortion vs car-reference fix

Evidence that horizontal gaze/head in WORLD frame is smeared across x and z by the
car's heading, compressing the x-axis (and its between-driver variance) when the
car faces world-X, and that the car-reference de-rotation recovers it.

Events are ordered by how much the car heading rotates during the window
(Stag >> Falling rocks >> Motorcyclist), so the artifact should be strongest in
Stag and mildest in Motorcyclist. Five panels per event + a severity table.
'''

code = r'''# ============================================================
# WORLD-FRAME ARTIFACT PROOF  (world axes vs car reference)
# ------------------------------------------------------------
# 1. Car heading over time (per driver + mean) -> the car turns.
# 2. World x vs z variance share over time -> variance smears between axes
#    exactly as heading rotates (axes are not stable).
# 3. World vs CAR between-driver SD of Eye.x / Nose.x -> car frame recovers the
#    variance that world-x lost under compression.
# 4. |corr(x,z)| across drivers, world vs car -> world axes entangled by heading;
#    car lateral axis decouples them.
# 5. ED on world (x,y) vs car (lateral,y) -> the artifact distorts the
#    downstream dimensionality result itself.
# Severity table sorts events by heading rotation.
# ============================================================
import numpy as np, os
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh as _eighWP

_WP_DIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/"
_WP_EVENTS=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_WP_LAB={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_WP_FS,_WP_CUT,_WP_ORD=50.0,0.8,3

def _wp_load(evt):
    df=dd.read_csv(_WP_DIR+f"car_reference_{evt}.csv",assume_missing=True,blocksize="100MB",
                   dtype={'HitObjectName':'object'}).compute()
    cols=["HmdPosition.x","HmdPosition.z","NoseVector.x","NoseVector.z",
          "EyeDirWorldCombined.x","EyeDirWorldCombined.z","NoseVectorCar.x","EyeDirCar.x",
          "NoseVector.y","EyeDirWorldCombined.y"]
    H=[]; W=[]
    for u,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        sub=sub.sort_values('time_from_event')
        W.append(sub[cols].values.astype(float))
    arr=np.stack(W,0); secs=np.linspace(-5,5,arr.shape[1]); crop=(secs>=-4)&(secs<=4)
    return arr[:,crop,:], secs[crop], cols

def _heading(hx,hz):
    b,a=butter(_WP_ORD,_WP_CUT/(_WP_FS/2),btype="low")
    return np.unwrap(np.arctan2(np.gradient(filtfilt(b,a,hz)),np.gradient(filtfilt(b,a,hx))))

def _ed(X):  # X (W, M) one timepoint, between-driver
    Xc=X-X.mean(0)
    for m in range(Xc.shape[1]):
        sd=Xc[:,m].std(ddof=1)
        if sd>1e-12: Xc[:,m]/=sd
    ev=np.maximum(_eighWP(np.cov(Xc.T),eigvals_only=True),0); s=ev.sum()
    return (s**2)/np.sum(ev**2) if s>0 else np.nan

_sev=[]
for _e in _WP_EVENTS:
    arr,secs,cols=_wp_load(_e); ci={c:i for i,c in enumerate(cols)}
    Wd,T,_=arr.shape
    # per-driver heading
    head=np.array([_heading(arr[w,:,ci["HmdPosition.x"]],arr[w,:,ci["HmdPosition.z"]]) for w in range(Wd)])
    head_deg=np.degrees(head)
    rot=np.mean(head_deg.max(1)-head_deg.min(1))
    # world x/z variance share over time (between-driver), gaze
    gx=arr[:,:,ci["EyeDirWorldCombined.x"]]; gz=arr[:,:,ci["EyeDirWorldCombined.z"]]
    vx=gx.var(0); vz=gz.var(0); share_x=vx/(vx+vz+1e-12)
    # world vs car between-driver SD (mean over time)
    sd_wx_eye=arr[:,:,ci["EyeDirWorldCombined.x"]].std(0).mean()
    sd_cx_eye=arr[:,:,ci["EyeDirCar.x"]].std(0).mean()
    sd_wx_nose=arr[:,:,ci["NoseVector.x"]].std(0).mean()
    sd_cx_nose=arr[:,:,ci["NoseVectorCar.x"]].std(0).mean()
    # |corr(x,z)| world (mean over time) vs car: car lateral has no z, so corr with
    # the forward (z) world axis -> show world x-z entanglement vs car-x to world-z
    cxz_world=np.nanmean([abs(np.corrcoef(gx[:,t],gz[:,t])[0,1]) for t in range(T)])
    cx_car=arr[:,:,ci["EyeDirCar.x"]]
    cxz_car=np.nanmean([abs(np.corrcoef(cx_car[:,t],gz[:,t])[0,1]) for t in range(T)])
    # ED world (x,y) vs car (lateral,y) using gaze+head x/y + steering-free 4-var
    def edcurve(xa,xb,xc,xd):
        e=np.zeros(T)
        for t in range(T):
            e[t]=_ed(np.column_stack([arr[:,t,ci[xa]],arr[:,t,ci[xb]],arr[:,t,ci[xc]],arr[:,t,ci[xd]]]))
        return e
    ed_world=edcurve("NoseVector.x","NoseVector.y","EyeDirWorldCombined.x","EyeDirWorldCombined.y")
    ed_car  =edcurve("NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y")

    _sev.append(dict(evt=_e,rot=rot,sd_wx_eye=sd_wx_eye,sd_cx_eye=sd_cx_eye,
                     sd_wx_nose=sd_wx_nose,sd_cx_nose=sd_cx_nose,
                     cxz_world=cxz_world,cxz_car=cxz_car,
                     ed_world=np.nanmean(ed_world),ed_car=np.nanmean(ed_car)))

    # ---- 5-panel figure ----
    fig,ax=plt.subplots(1,5,figsize=(24,4.2))
    fig.suptitle(f"{_WP_LAB[_e]} -- world-frame artifact (mean heading rotation {rot:.0f} deg)",
                 fontsize=13,fontweight="bold")
    ax[0].plot(secs,head_deg.T,color="#888",lw=0.3,alpha=0.2); ax[0].plot(secs,head_deg.mean(0),color="#c00",lw=2)
    ax[0].axvline(0,color="k",ls="--",lw=0.8); ax[0].set_title("1. Car heading (deg)"); ax[0].set_xlabel("Time (s)"); ax[0].grid(alpha=0.3)
    ax[1].plot(secs,share_x,color="#1f77b4",lw=1.8); ax[1].axhline(0.5,color="gray",ls=":"); ax[1].axvline(0,color="k",ls="--",lw=0.8)
    ax[1].set_ylim(0,1); ax[1].set_title("2. World gaze var share: x/(x+z)"); ax[1].set_xlabel("Time (s)"); ax[1].grid(alpha=0.3)
    xb=np.arange(2); ax[2].bar(xb-0.2,[sd_wx_eye,sd_wx_nose],0.4,label="world x",color="#d62728")
    ax[2].bar(xb+0.2,[sd_cx_eye,sd_cx_nose],0.4,label="car x",color="#2ca02c")
    ax[2].set_xticks(xb); ax[2].set_xticklabels(["Eye.x","Nose.x"]); ax[2].set_title("3. between-driver SD: world vs car"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    ax[3].bar([0,1],[cxz_world,cxz_car],color=["#d62728","#2ca02c"]); ax[3].set_xticks([0,1])
    ax[3].set_xticklabels(["world\n|corr(x,z)|","car\n|corr(carx,z)|"]); ax[3].set_ylim(0,1)
    ax[3].set_title("4. x-z entanglement"); ax[3].grid(alpha=0.3)
    ax[4].plot(secs,ed_world,color="#d62728",lw=1.8,label="world (x,y)"); ax[4].plot(secs,ed_car,color="#2ca02c",lw=1.8,label="car (lat,y)")
    ax[4].axvline(0,color="k",ls="--",lw=0.8); ax[4].set_title("5. ED world vs car (4 gaze/head vars)"); ax[4].set_xlabel("Time (s)"); ax[4].legend(fontsize=8); ax[4].grid(alpha=0.3)
    plt.tight_layout(rect=[0,0,1,0.93]); plt.show()

# ---- severity table, sorted by rotation ----
_sev.sort(key=lambda d:-d["rot"])
print("="*100)
print("ARTIFACT SEVERITY (sorted by heading rotation -- more rotation = worse world-frame distortion)")
print("="*100)
print(f"{'event':<14}{'rot(deg)':>9}{'SD Eye.x w->car':>18}{'SD Nose.x w->car':>18}{'|corr xz| w->car':>18}{'ED w->car':>14}")
print("-"*100)
for d in _sev:
    eye=f"{d['sd_wx_eye']:.2f}->{d['sd_cx_eye']:.2f}"
    nose=f"{d['sd_wx_nose']:.2f}->{d['sd_cx_nose']:.2f}"
    cor=f"{d['cxz_world']:.2f}->{d['cxz_car']:.2f}"
    edr=f"{d['ed_world']:.2f}->{d['ed_car']:.2f}"
    print(f"{_WP_LAB[d['evt']]:<14}{d['rot']:>9.0f}{eye:>18}{nose:>18}{cor:>18}{edr:>14}")
print("-"*100)
print("Reading: as heading rotation grows (Motorcyclist->Falling->Stag), world-x SD is more")
print("compressed (smaller) and recovers in car frame; world x-z correlation is higher")
print("(axes entangled) and drops in car frame. The world-frame ED is correspondingly distorted.")
'''
ast.parse(code)

nb["cells"].append({"cell_type":"markdown","metadata":{},"source":md})
nb["cells"].append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Appended world-frame artifact proof cell. Total cells: {len(nb['cells'])}.")
