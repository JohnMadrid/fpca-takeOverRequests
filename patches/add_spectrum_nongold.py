import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA_temp_16.6.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
already=any("EIGENVALUE SPECTRUM OVER TIME (NON-GOLD" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "non-gold spectrum cell present."

# insert right after the gold spectrum cell
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "EIGENVALUE SPECTRUM OVER TIME (GOLD" in "".join(c["source"]): ins=i
if ins is None: ins=len(nb["cells"])-1

code = r'''# ============================================================
# EIGENVALUE SPECTRUM OVER TIME (NON-GOLD: per-driver StandardScaler)
# ------------------------------------------------------------
# Same as the gold spectrum cell but with the obsolete recipe: per-driver
# StandardScaler over the 8s window, then per-timepoint cross-driver center +
# per-channel z. For comparison against the gold version.
# Outliers-removed car frame, 8s window. PC1 pre/post label-shuffle perm.
# ============================================================
import numpy as np, os
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scipy.linalg import eigh as _eighNG
from sklearn.preprocessing import StandardScaler

_NG_VARS=["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_NG_M=len(_NG_VARS)
_NG_EVENTS=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_NG_LAB={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_NG_DIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/outliers_removed_5features/"
_NG_COLORS=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
_NG_NPERM=5000; _NG_SEED=0

def _ng_ratios(evt):
    fp=_NG_DIR+f"car_reference_{evt}.csv"
    if not os.path.exists(fp): return None,None
    df=dd.read_csv(fp,assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()
    cs=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        raw=sub[_NG_VARS].values.astype(float); s0=np.linspace(-5,5,raw.shape[0]); raw=raw[(s0>=-4)&(s0<=4)]
        cs.append(StandardScaler().fit_transform(raw))          # per-driver z over 8s (NON-GOLD)
    arr=np.stack(cs,0); W,T,_=arr.shape; secs=np.linspace(-5,5,points_per_window); secs=secs[(secs>=-4)&(secs<=4)]
    ratios=np.full((_NG_M,T),np.nan)
    for t in range(T):
        sn=arr[:,t,:].copy(); sn-=sn.mean(0)
        for m in range(_NG_M):
            sd=sn[:,m].std(ddof=1)
            if sd>1e-12: sn[:,m]/=sd
        ev=np.sort(np.maximum(_eighNG(sn.T@sn/(W-1),eigvals_only=True),0))[::-1]
        tot=ev.sum()
        if tot>0: ratios[:,t]=ev/tot
    return secs,ratios

def _ng_perm(t,curve,n_perm=_NG_NPERM,seed=_NG_SEED):
    pre=t<0; post=t>=0; obs=np.nanmean(curve[post])-np.nanmean(curve[pre])
    idx=np.where(pre|post)[0]; vals=curve[idx]; nP=int(post.sum())
    rng=np.random.default_rng(seed); null=np.empty(n_perm); a=np.arange(len(vals))
    for p in range(n_perm):
        rng.shuffle(a); null[p]=np.nanmean(vals[a[:nP]])-np.nanmean(vals[a[nP:]])
    return obs,float(np.mean(null>=obs))

_ng={}
for _e in _NG_EVENTS:
    secs,r=_ng_ratios(_e)
    if secs is None: print(f"{_e}: not found"); continue
    _ng[_e]=(secs,r)

_ng_res={}
for _e in _NG_EVENTS:
    if _e not in _ng: continue
    secs,r=_ng[_e]
    fig,ax=plt.subplots(figsize=(12,4.5))
    for i in range(_NG_M): ax.plot(secs,r[i],color=_NG_COLORS[i],lw=2,alpha=0.9,label=f"PC{i+1}")
    obs,p=_ng_perm(secs,r[0]); _ng_res[_e]=(obs,p)
    ax.set_title(f"{_NG_LAB[_e]}  expl. variance (NON-GOLD: per-driver z)\nPC1 post-pre = {obs:+.3f}   p = {p:.3f}",fontsize=12)
    ax.set_ylim(0,1.0); ax.set_xlabel("Time (s)"); ax.set_ylabel("Explained variance ratio")
    ax.axvline(0,color="black",ls="--",lw=0.8,alpha=0.5); ax.grid(True,alpha=0.3); ax.minorticks_on()
    ax.grid(which="minor",ls=":",alpha=0.2); ax.legend(loc="upper right",fontsize=9)
    plt.tight_layout(); plt.show()

_evk=[e for e in _NG_EVENTS if e in _ng]; ref=_ng[_evk[0]][0]
def _al(t,c): return c if (len(t)==len(ref) and np.allclose(t,ref)) else np.interp(ref,t,c,left=np.nan,right=np.nan)
stack=np.array([[_al(_ng[e][0],_ng[e][1][i]) for e in _evk] for i in range(_NG_M)])
fig,ax=plt.subplots(figsize=(12,4.5))
for i in range(_NG_M):
    m=np.nanmean(stack[i],0); se=np.nanstd(stack[i],0,ddof=1)/np.sqrt(len(_evk))
    ax.plot(ref,m,color=_NG_COLORS[i],lw=2,alpha=0.9,label=f"PC{i+1}"); ax.fill_between(ref,m-se,m+se,color=_NG_COLORS[i],alpha=0.18,lw=0)
obsA,pA=_ng_perm(ref,np.nanmean(stack[0],0))
ax.set_title(f"Mean (+/- SE) expl. variance across {len(_evk)} events (NON-GOLD)\nPC1 post-pre = {obsA:+.3f}   p = {pA:.3f}",fontsize=12)
ax.set_ylim(0,1.0); ax.set_xlabel("Time (s)"); ax.set_ylabel("Explained variance ratio")
ax.axvline(0,color="black",ls="--",lw=0.8,alpha=0.5); ax.grid(True,alpha=0.3); ax.minorticks_on()
ax.grid(which="minor",ls=":",alpha=0.2); ax.legend(loc="upper right",fontsize=9)
plt.tight_layout(); plt.show()

print("\n"+"="*70)
print(f"PC1 pre vs post (NON-GOLD per-driver z, N_PERM={_NG_NPERM}, one-tailed post>pre)")
print("="*70)
print(f"{'Event':<20}{'post-pre':<12}{'p':<8}"); print("-"*40)
for _e in _evk: print(f"{_NG_LAB[_e]:<20}{_ng_res[_e][0]:+10.4f}   {_ng_res[_e][1]:.3f}")
print(f"{'Averaged curve':<20}{obsA:+10.4f}   {pA:.3f}")
'''
ast.parse(code)
nb["cells"].insert(ins+1, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted non-gold spectrum cell after {ins}. Total: {len(nb['cells'])}.")
