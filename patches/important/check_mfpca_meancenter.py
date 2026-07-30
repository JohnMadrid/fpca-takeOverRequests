"""Does the MFPCA post-onset ED RISE survive per-driver mean-centering?
Decomposition:
  (0) baseline      : current _mf_twostep_raw (cross-driver self-std then two-step)
  (1) demean-driver : subtract each driver's own per-channel TEMPORAL mean (removes
                      between-driver LOCATION shift; keeps trajectory SHAPE + amplitude)
  (2) demean+L2     : also divide each driver-channel by its own L2 norm (removes
                      amplitude too; keeps SHAPE pattern only)
If post-ED still > pre-ED under (1)/(2) -> the MFPCA rise is real shape divergence.
If it collapses -> the rise was a between-driver mean/location shift.
"""
import os, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import dask.dataframe as dd
from numpy.linalg import eigh as _eigh

points_per_window = 501
DATA = os.getcwd() + "/data/"
MODS = ["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
M = len(MODS)
EVENTS = ["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
LAB = {"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
PRE  = np.arange(50,250)
POST = np.arange(250,450)

def load_event(evt):
    fp = DATA + f"cleaned_data/data_segment/car_reference/car_reference_{evt}.csv"
    df = dd.read_csv(fp, assume_missing=True, blocksize="100MB").compute()
    curves=[]
    for _,sub in df.groupby("uid"):
        if len(sub)!=points_per_window: continue
        curves.append(sub.sort_values("time_from_event")[MODS].values.astype(float))
    return np.stack(curves,0)   # (W,501,M)

def ed(jev):
    jev=np.maximum(jev,0); s=jev.sum()
    return float((s**2)/np.sum(jev**2)) if s>0 else np.nan

def selfstd(a):
    flat=a.reshape(-1,a.shape[2]); mu=flat.mean(0); sd=flat.std(0,ddof=1); sd[sd<1e-12]=1.0
    return (a-mu)/sd

def twostep(a):
    """Same as _mf_twostep_raw: cross-driver self-std, per-channel lossless PCA, stack, joint PCA -> ED."""
    a=selfstd(a.astype(float)); stacked=[]
    for ci in range(M):
        Xc=a[:,:,ci]-a[:,:,ci].mean(0)
        ev,evec=_eigh(np.cov(Xc.T)); o=np.argsort(ev)[::-1]; ev=np.maximum(ev[o],0); evec=evec[:,o]
        nk=max(1,int(np.sum(ev>1e-12)))
        stacked.append(Xc@evec[:,:nk])
    Z=np.hstack(stacked); Z=Z-Z.mean(0)
    jev=np.maximum(_eigh(np.cov(Z.T))[0],0)
    return ed(jev)

def demean_driver(a):
    """Subtract each driver's own per-channel temporal mean. a:(W,T,M)."""
    return a - a.mean(axis=1, keepdims=True)

def demean_l2(a):
    """Per-driver demean then per-driver-channel L2 normalise."""
    b = a - a.mean(axis=1, keepdims=True)
    nrm = np.sqrt((b**2).sum(axis=1, keepdims=True)); nrm[nrm<1e-12]=1.0
    return b / nrm

print(f"{'event':<16}{'window':<6}{'(0)baseline':>13}{'(1)demean':>11}{'(2)demean+L2':>14}")
print("-"*60)
rows={}
for evt in EVENTS:
    arr=load_event(evt)
    pre=arr[:,PRE,:]; post=arr[:,POST,:]
    res={}
    for tag,win in [("pre",pre),("post",post)]:
        e0=twostep(win)
        e1=twostep(demean_driver(win))
        e2=twostep(demean_l2(win))
        res[tag]=(e0,e1,e2)
        print(f"{LAB[evt]:<16}{tag:<6}{e0:>13.2f}{e1:>11.2f}{e2:>14.2f}")
    rows[evt]=res
    # deltas
    d=lambda k:res['post'][k]-res['pre'][k]
    print(f"{'':16}{'Δ post-pre':<6}{d(0):>+13.2f}{d(1):>+11.2f}{d(2):>+14.2f}")
    print("-"*60)

print("\nReading:")
print(" - If Δ stays positive under (1)/(2): post rise = real trajectory-SHAPE divergence.")
print(" - If Δ goes ~0 or negative under (1): the rise was a between-driver LOCATION shift.")
print(" - (2) isolates pure shape (amplitude removed); compare (1) vs (2) for amplitude role.")
