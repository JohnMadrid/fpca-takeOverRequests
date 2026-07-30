import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA_temp_16.6.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
already=any("EIGENVALUE SPECTRUM OVER TIME (GOLD" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "spectrum cell present."

# insert after the gold ED cell
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "ED PRE vs POST -- GOLD STANDARD" in "".join(c["source"]): ins=i
if ins is None: ins=len(nb["cells"])-1

code = r'''# ============================================================
# EIGENVALUE SPECTRUM OVER TIME (GOLD recipe) -- per event + mean, with PC1 perm
# ------------------------------------------------------------
# Per timepoint: gold-recipe (C + pooled per-channel scale) 5x5 cross-driver
# covariance -> eigenvalues -> normalised explained-variance ratios PC1..PC5.
# Plots ratio-over-time per event + mean+/-SE across events.
# PC1 pre-vs-post label-shuffle permutation (one-tailed post>pre), as in the
# pca_lda_5var "PCA Explained Variance Over Time" cell.
# Outliers-removed car frame, 8s window.
# ============================================================
import numpy as np, os
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scipy.linalg import eigh as _eighSP

_SP_VARS=["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_SP_M=len(_SP_VARS)
_SP_EVENTS=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_SP_LAB={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_SP_DIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/outliers_removed_5features/"
_PC_COLORS=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
_N_PERM=5000; _PERM_SEED=0

def _sp_ratios(evt):
    """Returns (secs, ratios[5,T]) gold-recipe normalised explained-variance over time."""
    fp=_SP_DIR+f"car_reference_{evt}.csv"
    if not os.path.exists(fp): return None,None
    df=dd.read_csv(fp,assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()
    cs=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        raw=sub[_SP_VARS].values.astype(float); s0=np.linspace(-5,5,raw.shape[0]); raw=raw[(s0>=-4)&(s0<=4)]
        cs.append(raw)
    arr=np.stack(cs,0).astype(float)
    arr=arr-arr.mean(axis=1,keepdims=True)                       # C
    flat=arr.reshape(-1,_SP_M); psd=flat.std(0,ddof=1); psd[psd<1e-12]=1.0
    W,T,_=arr.shape; secs=np.linspace(-5,5,points_per_window); secs=secs[(secs>=-4)&(secs<=4)]
    ratios=np.full((_SP_M,T),np.nan)
    for t in range(T):
        sn=arr[:,t,:].copy(); sn=sn-sn.mean(0); sn=sn/psd                 # pooled scale
        ev=np.sort(np.maximum(_eighSP(sn.T@sn/(W-1),eigvals_only=True),0))[::-1]
        tot=ev.sum()
        if tot>0: ratios[:,t]=ev/tot
    return secs,ratios

def _perm_prepost(t,curve,n_perm=_N_PERM,seed=_PERM_SEED):
    pre=t<0; post=t>=0
    obs=np.nanmean(curve[post])-np.nanmean(curve[pre])
    idx=np.where(pre|post)[0]; vals=curve[idx]; nP=int(post.sum())
    rng=np.random.default_rng(seed); null=np.empty(n_perm); a=np.arange(len(vals))
    for p in range(n_perm):
        rng.shuffle(a); null[p]=np.nanmean(vals[a[:nP]])-np.nanmean(vals[a[nP:]])
    return obs,float(np.mean(null>=obs))   # one-tailed post>pre

# ---- compute per event ----
_sp={}
for _e in _SP_EVENTS:
    secs,r=_sp_ratios(_e)
    if secs is None: print(f"{_e}: not found"); continue
    _sp[_e]=(secs,r)

# ---- per-event figures ----
_pc1res={}
for _e in _SP_EVENTS:
    if _e not in _sp: continue
    secs,r=_sp[_e]
    fig,ax=plt.subplots(figsize=(12,4.5))
    for i in range(_SP_M):
        ax.plot(secs,r[i],color=_PC_COLORS[i],lw=2,alpha=0.9,label=f"PC{i+1}")
    obs,p=_perm_prepost(secs,r[0]); _pc1res[_e]=(obs,p)
    ax.set_title(f"{_SP_LAB[_e]}  normalised explained variance\nPC1 post-pre = {obs:+.3f}   p = {p:.3f}",fontsize=12)
    ax.set_ylim(0,0.6); ax.set_xlabel("Time (s)"); ax.set_ylabel("Explained variance ratio")
    ax.axvline(0,color="black",ls="--",lw=0.8,alpha=0.5); ax.grid(True,alpha=0.3); ax.minorticks_on()
    ax.grid(which="minor",ls=":",alpha=0.2); ax.legend(loc="upper right",fontsize=9)
    plt.tight_layout(); plt.show()

# ---- mean +/- SE across events ----
_evk=[e for e in _SP_EVENTS if e in _sp]
ref=_sp[_evk[0]][0]
def _align(t,c): return c if (len(t)==len(ref) and np.allclose(t,ref)) else np.interp(ref,t,c,left=np.nan,right=np.nan)
stack=np.array([[_align(_sp[e][0],_sp[e][1][i]) for e in _evk] for i in range(_SP_M)])  # (PC,event,T)
fig,ax=plt.subplots(figsize=(12,4.5))
for i in range(_SP_M):
    m=np.nanmean(stack[i],0); se=np.nanstd(stack[i],0,ddof=1)/np.sqrt(len(_evk))
    ax.plot(ref,m,color=_PC_COLORS[i],lw=2,alpha=0.9,label=f"PC{i+1}")
    ax.fill_between(ref,m-se,m+se,color=_PC_COLORS[i],alpha=0.18,lw=0)
pc1mean=np.nanmean(stack[0],0); obsA,pA=_perm_prepost(ref,pc1mean)
ax.set_title(f"Mean (+/- SE) normalised explained variance across {len(_evk)} events\nPC1 post-pre = {obsA:+.3f}   p = {pA:.3f}",fontsize=12)
ax.set_ylim(0,0.6); ax.set_xlabel("Time (s)"); ax.set_ylabel("Explained variance ratio")
ax.axvline(0,color="black",ls="--",lw=0.8,alpha=0.5); ax.grid(True,alpha=0.3); ax.minorticks_on()
ax.grid(which="minor",ls=":",alpha=0.2); ax.legend(loc="upper right",fontsize=9)
plt.tight_layout(); plt.show()

print("\n"+"="*70)
print(f"PC1 pre vs post label-shuffle (N_PERM={_N_PERM}, one-tailed post>pre), gold recipe")
print("="*70)
print(f"{'Event':<20}{'post-pre':<12}{'p':<8}")
print("-"*40)
for _e in _evk: print(f"{_SP_LAB[_e]:<20}{_pc1res[_e][0]:+10.4f}   {_pc1res[_e][1]:.3f}")
print(f"{'Averaged curve':<20}{obsA:+10.4f}   {pA:.3f}")
'''
ast.parse(code)
nb["cells"].insert(ins+1, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted spectrum-over-time cell after {ins}. Total: {len(nb['cells'])}.")
