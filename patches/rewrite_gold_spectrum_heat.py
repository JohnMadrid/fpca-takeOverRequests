import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA_temp_16.6.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "EIGENVALUE SPECTRUM OVER TIME (GOLD" in "".join(c["source"]): break
ci=i

code = r'''# ============================================================
# EIGENVALUE SPECTRUM OVER TIME (GOLD) + PC1 contribution heatmaps
# ------------------------------------------------------------
# Gold recipe (C + pooled per-channel scale), outliers-removed car frame, 8s.
# Per event:
#   (1) PC1..PC5 explained-variance ratio over time, with mean-RT line.
#   (2) Continuous heatmap: PC1 contribution per channel over time (cos^2 =
#       squared PC1 loading; sums to 1 per timepoint).
#   (3) 5x3 heatmap: PC1 cos^2 averaged over Baseline / TOR / Action.
# Plus the across-event mean spectrum, and PC1 pre/post label-shuffle perm.
# ============================================================
import numpy as np, os
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scipy.linalg import eigh as _eighSP

_SP_VARS=["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_SP_VLAB=["Head.x","Head.y","Eye.x","Eye.y","Steering"]
_SP_M=len(_SP_VARS)
_SP_EVENTS=["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_SP_LAB={"StagEventNew":"Stag crossing","FallingRocksEventNew":"Falling rocks","MotorcyclistEvent":"Motorcyclist"}
_SP_DIR=os.getcwd()+"/data/cleaned_data/data_segment/car_reference/outliers_removed_5features/"
_PC_COLORS=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
_N_PERM=5000; _PERM_SEED=0

# mean RT per event
import pandas as pd
_rtcsv=pd.read_csv(os.getcwd()+"/reaction_times.csv")
_rtcsv["RT_Steering"]=pd.to_numeric(_rtcsv["RT_Steering"],errors="coerce")
_rtcsv=_rtcsv[(_rtcsv["Unavailable_Steering"].astype(str).str.lower()!="true") & _rtcsv["RT_Steering"].notna()]
def _mrt(e):
    s=_rtcsv[_rtcsv["EventName"]==e]; return float(s["RT_Steering"].mean()) if len(s) else np.nan

def _sp_compute(evt):
    """Returns secs, ratios[5,T], pc1cos2[5,T] (gold recipe)."""
    fp=_SP_DIR+f"car_reference_{evt}.csv"
    if not os.path.exists(fp): return None,None,None
    df=dd.read_csv(fp,assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()
    cs=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        raw=sub[_SP_VARS].values.astype(float); s0=np.linspace(-5,5,raw.shape[0]); raw=raw[(s0>=-4)&(s0<=4)]
        cs.append(raw)
    arr=np.stack(cs,0).astype(float)
    arr=arr-arr.mean(axis=1,keepdims=True)
    flat=arr.reshape(-1,_SP_M); psd=flat.std(0,ddof=1); psd[psd<1e-12]=1.0
    W,T,_=arr.shape; secs=np.linspace(-5,5,points_per_window); secs=secs[(secs>=-4)&(secs<=4)]
    ratios=np.full((_SP_M,T),np.nan); pc1=np.full((_SP_M,T),np.nan)
    for t in range(T):
        sn=arr[:,t,:].copy(); sn=sn-sn.mean(0); sn=sn/psd
        evals,evecs=_eighSP(sn.T@sn/(W-1))
        o=np.argsort(evals)[::-1]; evals=np.maximum(evals[o],0); evecs=evecs[:,o]
        tot=evals.sum()
        if tot>0:
            ratios[:,t]=evals/tot
            v=evecs[:,0]; pc1[:,t]=v**2          # cos^2, sums to 1
    return secs,ratios,pc1

def _perm_prepost(t,curve,n_perm=_N_PERM,seed=_PERM_SEED):
    pre=t<0; post=t>=0; obs=np.nanmean(curve[post])-np.nanmean(curve[pre])
    idx=np.where(pre|post)[0]; vals=curve[idx]; nP=int(post.sum())
    rng=np.random.default_rng(seed); null=np.empty(n_perm); a=np.arange(len(vals))
    for p in range(n_perm):
        rng.shuffle(a); null[p]=np.nanmean(vals[a[:nP]])-np.nanmean(vals[a[nP:]])
    return obs,float(np.mean(null>=obs))

def _phase_avg(secs,mat,rt):
    B=(secs>=-4)&(secs<0); Tr=(secs>=0)&(secs<rt); A=(secs>=rt)&(secs<=4)
    return np.column_stack([mat[:,B].mean(1),mat[:,Tr].mean(1),mat[:,A].mean(1)])  # (5,3)

_sp={}
for _e in _SP_EVENTS:
    secs,r,pc1=_sp_compute(_e)
    if secs is None: print(f"{_e}: not found"); continue
    _sp[_e]=(secs,r,pc1,_mrt(_e))

# ---- per-event stacked figure: spectrum + continuous heatmap + 3-phase heatmap ----
_pc1res={}
for _e in _SP_EVENTS:
    if _e not in _sp: continue
    secs,r,pc1,rt=_sp[_e]
    fig=plt.figure(figsize=(12,9))
    gs=fig.add_gridspec(3,1,height_ratios=[2.0,1.3,1.1],hspace=0.42)
    # (1) spectrum
    ax0=fig.add_subplot(gs[0])
    for k in range(_SP_M): ax0.plot(secs,r[k],color=_PC_COLORS[k],lw=2,alpha=0.9,label=f"PC{k+1}")
    obs,p=_perm_prepost(secs,r[0]); _pc1res[_e]=(obs,p)
    ax0.axvline(0,color="black",ls="--",lw=0.8,alpha=0.5)
    if np.isfinite(rt): ax0.axvline(rt,color="purple",ls=":",lw=1.6,label=f"mean RT={rt:.2f}s")
    ax0.set_ylim(0,1); ax0.set_ylabel("Explained variance ratio")
    ax0.set_title(f"{_SP_LAB[_e]}  normalised explained variance (gold)\nPC1 post-pre = {obs:+.3f}   p = {p:.3f}",fontsize=12)
    ax0.grid(True,alpha=0.3); ax0.legend(loc="upper right",fontsize=9)
    # (2) continuous PC1 cos2 heatmap
    ax1=fig.add_subplot(gs[1])
    im=ax1.imshow(pc1,aspect="auto",origin="lower",extent=[secs[0],secs[-1],-0.5,_SP_M-0.5],
                  cmap="magma",vmin=0,vmax=1)
    ax1.axvline(0,color="white",ls="--",lw=0.8,alpha=0.7)
    if np.isfinite(rt): ax1.axvline(rt,color="cyan",ls=":",lw=1.4)
    ax1.set_yticks(range(_SP_M)); ax1.set_yticklabels(_SP_VLAB,fontsize=9)
    ax1.set_xlabel("Time (s)"); ax1.set_title("PC1 contribution per channel over time (cos²)",fontsize=10)
    fig.colorbar(im,ax=ax1,fraction=0.025,pad=0.01,label="cos²")
    # (3) 3-phase averaged heatmap
    ax2=fig.add_subplot(gs[2])
    pa=_phase_avg(secs,pc1,rt)   # (5,3)
    im2=ax2.imshow(pa,aspect="auto",cmap="magma",vmin=0,vmax=1)
    ax2.set_xticks([0,1,2]); ax2.set_xticklabels(["Baseline","TOR","Action"],fontsize=9)
    ax2.set_yticks(range(_SP_M)); ax2.set_yticklabels(_SP_VLAB,fontsize=9)
    ax2.set_title("PC1 cos² averaged per phase",fontsize=10)
    for yy in range(_SP_M):
        for xx in range(3):
            ax2.text(xx,yy,f"{pa[yy,xx]:.2f}",ha="center",va="center",
                     color="white" if pa[yy,xx]<0.5 else "black",fontsize=8)
    fig.colorbar(im2,ax=ax2,fraction=0.025,pad=0.01,label="cos²")
    plt.show()

# ---- across-event mean spectrum (with mean RT line) ----
_evk=[e for e in _SP_EVENTS if e in _sp]; ref=_sp[_evk[0]][0]
def _al(t,c): return c if (len(t)==len(ref) and np.allclose(t,ref)) else np.interp(ref,t,c,left=np.nan,right=np.nan)
stackR=np.array([[_al(_sp[e][0],_sp[e][1][k]) for e in _evk] for k in range(_SP_M)])
stackP=np.array([[_al(_sp[e][0],_sp[e][2][k]) for e in _evk] for k in range(_SP_M)])
rt_avg=np.nanmean([_sp[e][3] for e in _evk])
fig=plt.figure(figsize=(12,9)); gs=fig.add_gridspec(3,1,height_ratios=[2.0,1.3,1.1],hspace=0.42)
ax0=fig.add_subplot(gs[0])
for k in range(_SP_M):
    m=np.nanmean(stackR[k],0); se=np.nanstd(stackR[k],0,ddof=1)/np.sqrt(len(_evk))
    ax0.plot(ref,m,color=_PC_COLORS[k],lw=2,alpha=0.9,label=f"PC{k+1}"); ax0.fill_between(ref,m-se,m+se,color=_PC_COLORS[k],alpha=0.18,lw=0)
obsA,pA=_perm_prepost(ref,np.nanmean(stackR[0],0))
ax0.axvline(0,color="black",ls="--",lw=0.8,alpha=0.5); ax0.axvline(rt_avg,color="purple",ls=":",lw=1.6,label=f"mean RT={rt_avg:.2f}s")
ax0.set_ylim(0,1); ax0.set_ylabel("Explained variance ratio")
ax0.set_title(f"Mean (+/- SE) explained variance across {len(_evk)} events (gold)\nPC1 post-pre = {obsA:+.3f}   p = {pA:.3f}",fontsize=12)
ax0.grid(True,alpha=0.3); ax0.legend(loc="upper right",fontsize=9)
pc1avg=np.nanmean(stackP,1)   # (5,T)
ax1=fig.add_subplot(gs[1])
im=ax1.imshow(pc1avg,aspect="auto",origin="lower",extent=[ref[0],ref[-1],-0.5,_SP_M-0.5],cmap="magma",vmin=0,vmax=1)
ax1.axvline(0,color="white",ls="--",lw=0.8,alpha=0.7); ax1.axvline(rt_avg,color="cyan",ls=":",lw=1.4)
ax1.set_yticks(range(_SP_M)); ax1.set_yticklabels(_SP_VLAB,fontsize=9); ax1.set_xlabel("Time (s)")
ax1.set_title("PC1 contribution per channel over time (cos², avg events)",fontsize=10)
fig.colorbar(im,ax=ax1,fraction=0.025,pad=0.01,label="cos²")
ax2=fig.add_subplot(gs[2])
paA=_phase_avg(ref,pc1avg,rt_avg)
im2=ax2.imshow(paA,aspect="auto",cmap="magma",vmin=0,vmax=1)
ax2.set_xticks([0,1,2]); ax2.set_xticklabels(["Baseline","TOR","Action"],fontsize=9)
ax2.set_yticks(range(_SP_M)); ax2.set_yticklabels(_SP_VLAB,fontsize=9)
ax2.set_title("PC1 cos² averaged per phase (avg events)",fontsize=10)
for yy in range(_SP_M):
    for xx in range(3):
        ax2.text(xx,yy,f"{paA[yy,xx]:.2f}",ha="center",va="center",color="white" if paA[yy,xx]<0.5 else "black",fontsize=8)
fig.colorbar(im2,ax=ax2,fraction=0.025,pad=0.01,label="cos²")
plt.show()

print("\n"+"="*70)
print(f"PC1 pre vs post label-shuffle (N_PERM={_N_PERM}, one-tailed post>pre), gold")
print("="*70)
print(f"{'Event':<20}{'post-pre':<12}{'p':<8}"); print("-"*40)
for _e in _evk: print(f"{_SP_LAB[_e]:<20}{_pc1res[_e][0]:+10.4f}   {_pc1res[_e][1]:.3f}")
print(f"{'Averaged curve':<20}{obsA:+10.4f}   {pA:.3f}")
'''
ast.parse(code)
nb["cells"][ci]["source"]=code
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {ci}: gold spectrum + RT line + PC1 cos2 continuous & 3-phase heatmaps.")
