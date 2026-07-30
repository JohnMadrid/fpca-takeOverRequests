import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already = any("GOLD ED -- PERMUTATION TEST" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source",""))
              for c in nb["cells"])
assert not already, "gold perm-test cell already present."

ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "ED PRE vs POST -- GOLD STANDARD" in "".join(c["source"]): ins=i
assert ins is not None, "gold ED cell not found"

md = r'''### Permutation test: gold-standard ED drop (post < pre)

Same permutation method as the PCA car-frame permutation cell, on the gold recipe
(per-driver center C + pooled per-channel scale, 8s window). Null: per-driver
pre/post flip via time-axis reversal. Statistic: mean(post ED) - mean(pre ED)
(negative = post dips below pre). One-tailed p; per event + combined.
'''

code = r'''# ============================================================
# GOLD ED -- PERMUTATION TEST (C + pooled scale, 8s window, car frame)
# ------------------------------------------------------------
# Null: per-participant pre/post flip via time-axis reversal (preserves
# within-participant covariance, destroys time-locked structure).
# Statistic: mean(post ED) - mean(pre ED); one-tailed p (post below pre).
# Per event + combined (pooled participants; ED averaged across events).
# Self-contained. Gold recipe matches the gold ED pre/post cell above.
# ============================================================
import numpy as np, os
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scipy.linalg import eigh as _eighGP

_GP_B = 5000
_GP_VARS=["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_GP_EVENTS=["StagEventNew","MotorcyclistEvent","FallingRocksEventNew"]
_GP_LAB={"StagEventNew":"Stag crossing","MotorcyclistEvent":"Motorcyclist","FallingRocksEventNew":"Falling rocks"}
_GP_COL={"StagEventNew":"steelblue","MotorcyclistEvent":"#d62728","FallingRocksEventNew":"#2ca02c"}
_GP_DIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/"

def _gp_load(evt):
    """RAW per-driver curves, 8s crop. Returns (W,T,M), secs. Scaling deferred."""
    fp=_GP_DIR+f"car_reference_{evt}.csv"
    if not os.path.exists(fp): return None,None
    df=dd.read_csv(fp,assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()
    curves=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        _raw=sub.sort_values('time_from_event')[_GP_VARS].values.astype(float)
        _s0=np.linspace(-5,5,_raw.shape[0]); _raw=_raw[(_s0>=-4)&(_s0<=4)]
        curves.append(_raw)
    arr=np.stack(curves,0)
    secs=np.linspace(-5,5,points_per_window); secs=secs[(secs>=-4)&(secs<=4)]
    return arr, secs

def _gp_ed_curve(arr):
    """GOLD recipe: per-driver center (C) + pooled per-channel scale, then ED(t)."""
    A=arr.astype(float)-arr.astype(float).mean(axis=1,keepdims=True)        # C
    flat=A.reshape(-1,A.shape[2]); psd=flat.std(0,ddof=1); psd[psd<1e-12]=1.0
    W,T,M=A.shape; ed=np.zeros(T)
    for t in range(T):
        snap=A[:,t,:].copy(); snap=snap-snap.mean(0); snap=snap/psd
        ev=np.maximum(_eighGP(snap.T@snap/(W-1),eigvals_only=True),0); s=ev.sum()
        ed[t]=(s**2)/np.sum(ev**2) if s>0 else np.nan
    return ed

def _gp_perm(arr, secs, n_perm, rng, label):
    pre=secs<0; post=secs>=0; W=arr.shape[0]
    obs_ed=_gp_ed_curve(arr); obs=np.mean(obs_ed[post])-np.mean(obs_ed[pre])
    null=np.empty(n_perm)
    for p in range(n_perm):
        if p%500==0: print(f"  {label}: perm {p}/{n_perm}")
        ap=arr.copy(); flip=rng.random(W)<0.5
        ap[flip,:,:]=ap[flip][:,::-1,:]
        ed_p=_gp_ed_curve(ap)
        null[p]=np.mean(ed_p[post])-np.mean(ed_p[pre])
    return obs_ed, obs, null, float(np.mean(null<=obs))

def _gp_plot(secs, obs_ed, obs, null, p, color, title):
    pre=secs<0; post=secs>=0; pm=np.mean(obs_ed[pre])
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    ax=axes[0]; below=post&(obs_ed<pm)
    ax.plot(secs,obs_ed,color=color,lw=1.5,label='Observed ED')
    ax.fill_between(secs,obs_ed,pm,where=below,alpha=0.25,color='tomato',
                    label=f'Below pre-onset mean ({pm:.2f})')
    ax.axhline(pm,color='tomato',lw=1,ls='--',label=f'Pre-onset mean = {pm:.2f}')
    ax.axvline(0,color='gray',ls='--',lw=0.9,alpha=0.7)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Effective Dimensionality')
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    ax2=axes[1]
    ax2.hist(null,bins=40,color='gray',alpha=0.7,edgecolor='none')
    ax2.axvline(obs,color=color,lw=2,label=f'Observed = {obs:.4f}\np = {p:.3f}')
    ax2.axvline(0,color='black',lw=0.8,ls='--',alpha=0.5)
    ax2.set_xlabel('mean(post ED) - mean(pre ED)'); ax2.set_ylabel('Count')
    ax2.set_title(f'Null distribution  (n={len(null)} perms)')
    ax2.legend(fontsize=9); ax2.grid(True,alpha=0.3)
    fig.tight_layout(); plt.show()
    print(f"{title}")
    print(f"  Pre-onset mean ED        : {pm:.4f}")
    print(f"  Post-onset mean ED       : {np.mean(obs_ed[post]):.4f}")
    print(f"  Test statistic (post-pre): {obs:+.4f}")
    print(f"  p-value (one-tailed)     : {p:.3f}\\n")

_gp_arrs=[]; _gp_secs=None
for _e in _GP_EVENTS:
    print(f"Loading {_e} (gold)...")
    arr,secs=_gp_load(_e)
    if arr is None: print("  skipped"); continue
    print(f"  {arr.shape[0]} participants")
    _gp_arrs.append(arr); _gp_secs=secs
    oe,ob,nl,pv=_gp_perm(arr,secs,_GP_B,np.random.default_rng(7),_GP_LAB[_e])
    _gp_plot(secs,oe,ob,nl,pv,_GP_COL[_e],f"{_GP_LAB[_e]} -- gold ED")

# combined: pool participants
if len(_gp_arrs)>1:
    print("Combined (all events pooled)...")
    ac=np.concatenate(_gp_arrs,0)
    oe,ob,nl,pv=_gp_perm(ac,_gp_secs,_GP_B,np.random.default_rng(7),"Combined")
    _gp_plot(_gp_secs,oe,ob,nl,pv,'#555555',"Combined (all events pooled) -- gold ED")

# combined: ED averaged across events
if len(_gp_arrs)>1:
    print("Combined (ED averaged across events)...")
    oe_each=[]; nl_each=[]
    for ae in _gp_arrs:
        oe,ob,nl,_=_gp_perm(ae,_gp_secs,_GP_B,np.random.default_rng(7),"avg-loop")
        oe_each.append(oe); nl_each.append(nl)
    oe_avg=np.mean(np.stack(oe_each,0),0); nl_avg=np.mean(np.stack(nl_each,0),0)
    pre=_gp_secs<0; post=_gp_secs>=0
    ob_avg=np.mean(oe_avg[post])-np.mean(oe_avg[pre]); pv_avg=float(np.mean(nl_avg<=ob_avg))
    _gp_plot(_gp_secs,oe_avg,ob_avg,nl_avg,pv_avg,'#555555',"Combined (ED averaged) -- gold ED")
'''
ast.parse(code)

nb["cells"].insert(ins+1, {"cell_type":"markdown","metadata":{},"source":md})
nb["cells"].insert(ins+2, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted gold ED perm-test cell after {ins}. Total cells: {len(nb['cells'])}.")
