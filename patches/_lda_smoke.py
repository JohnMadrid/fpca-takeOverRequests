points_per_window=501
# ============================================================
# TFCE-LDA (CAR FRAME): Base-vs-Full & Success-vs-Failure, balanced vs unbalanced
# ------------------------------------------------------------
# Method (identical to pca_lda_5var cheap-TFCE): per-timepoint balanced-subsample
# LDA, z=(acc-chance)/sqrt(chance*(1-chance)/n), 1D TFCE, B label-perms -> null
# TFCE max -> one-tailed p/timepoint. Car-frame vars, raw (no operation C).
#
# Analyses:
#  (1) Base vs Full   : unbalanced (all)  |  balanced (equal succ/fail per arm)
#  (2) Succ vs Fail   : unbalanced (all)  |  balanced (equal condition mix per arm)
#  (3) Succ vs Fail   : Base only
#  (4) Succ vs Fail   : Full only
# Balanced = stratify out the OTHER factor; subsample equal n per stratum per arm.
# ============================================================
import numpy as np, warnings, os
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    def _tqdm(it, **kw): return it

_TFCE_B = 100
_TFCE_E, _TFCE_H, _TFCE_DT_STEPS, _TFCE_ALPHA = 0.5, 2.0, 100, 0.05
_BAL_SEED = 0

_CF_VARS = ["NoseVectorCar.x","NoseVector.y","EyeDirCar.x","EyeDirWorldCombined.y","SteeringInput"]
_CF_EVENTS = ["StagEventNew","FallingRocksEventNew","MotorcyclistEvent"]
_CF_DIR = os.getcwd()+"/data/cleaned_data/data_segment/car_reference/"

def _cf_smooth(x,k=5):
    x=np.asarray(x,float); out=x.copy()
    for i in range(len(x)):
        seg=x[max(0,i-k//2):min(len(x),i+k//2+1)]; seg=seg[~np.isnan(seg)]
        if len(seg): out[i]=seg.mean()
    return out

def _tfce_1d(stat,e=_TFCE_E,h=_TFCE_H,n_steps=_TFCE_DT_STEPS):
    s=np.where(stat>0,stat,0.0)
    if not np.any(s>0): return np.zeros_like(s)
    mx=s.max(); dh=mx/n_steps; out=np.zeros_like(s)
    for hi in np.arange(dh,mx+dh,dh):
        b=s>=hi
        if not np.any(b): continue
        d=np.diff(np.concatenate(([0],b.astype(int),[0])))
        for st,en in zip(np.where(d==1)[0],np.where(d==-1)[0]):
            out[st:en]+=((en-st)**e)*(hi**h)*dh
    return out

def _cf_load(evt):
    fp=_CF_DIR+f"car_reference_{evt}.csv"
    if not os.path.exists(fp): print(f"  {evt}: not found"); return None
    return dd.read_csv(fp,assume_missing=True,blocksize="100MB",dtype={'HitObjectName':'object'}).compute()

def _build(df,vars_):
    def fv(s): s=s.dropna(); return s.iloc[0] if len(s) else np.nan
    suc=df.groupby("uid")["SuccessfulCompletionState"].apply(fv).to_dict()
    wrn=df.groupby("uid")["ExperimentalCondition"].apply(fv).to_dict()
    dw=df[df["time_from_event"].between(-4.0,4.0)]
    ts=np.sort(dw["time_from_event"].unique())
    uids=sorted([u for u in dw["uid"].unique()
                 if suc.get(u) in (0.0,1.0)
                 and wrn.get(u) in ["BaseCondition","AudioOnly","HUDOnly","FullLoopAR"]])
    ui={u:i for i,u in enumerate(uids)}
    X=np.full((len(uids),len(ts),len(vars_)),np.nan)
    for _,row in dw[["uid","time_from_event"]+vars_].iterrows():
        u=row["uid"]
        if u not in ui: continue
        ti=np.searchsorted(ts,row["time_from_event"]); v=row[vars_].values.astype(float)
        if np.any(np.isnan(v)): continue
        X[ui[u],ti]=v
    yw=np.array([wrn[u] for u in uids])
    yo=np.array([int(suc[u]) for u in uids])
    return ts,X,yw,yo

# ---- per-timepoint balanced-subsample LDA ----
def _lda_acc_at_t(Xt,y,n_min,rng):
    cls=np.unique(y); idx=[]
    for c in cls:
        ci=np.where(y==c)[0]
        if len(ci)<n_min: return np.nan,0
        idx.extend(rng.choice(ci,n_min,replace=False).tolist())
    idx=np.array(idx); Xb=Xt[idx]; yb=y[idx]
    keep=~np.any(np.isnan(Xb),axis=1)
    if keep.sum()<4: return np.nan,0
    Xb=Xb[keep]; yb=yb[keep]
    if len(np.unique(yb))<2: return np.nan,0
    correct=tot=0
    skf=StratifiedKFold(n_splits=min(5,len(Xb)//2),shuffle=True,random_state=0)
    try:
        for tr,te in skf.split(Xb,yb):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lda=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"); lda.fit(Xb[tr],yb[tr])
                pred=lda.predict(Xb[te])
            correct+=int((pred==yb[te]).sum()); tot+=len(yb[te])
        return (correct/tot if tot else np.nan), tot
    except Exception:
        return np.nan,0

def _acc_tc(X,y,n_min,seed=0):
    rng=np.random.default_rng(seed); acc=np.full(X.shape[1],np.nan); nt=0
    for ti in range(X.shape[1]):
        a,n=_lda_acc_at_t(X[:,ti,:],y,n_min,rng); acc[ti]=a; nt=max(nt,n)
    return acc,nt

def _z(acc,ch,n):
    se=np.sqrt(max(ch*(1-ch)/max(n,1),1e-12)); return (acc-ch)/se
def _pval(obs,nm):
    p=np.array([np.mean(nm>=s) if not np.isnan(s) else np.nan for s in obs])
    return np.clip(p,1.0/(len(nm)+1),1.0)
def _clusters(sig):
    d=np.diff(np.concatenate(([0],sig.astype(int),[0])))
    return list(zip(np.where(d==1)[0],np.where(d==-1)[0]))
def _vlab(v):
    return {"NoseVectorCar.x":"Head.x","NoseVector.y":"Head.y","EyeDirCar.x":"Eye.x",
            "EyeDirWorldCombined.y":"Eye.y","SteeringInput":"Steering"}.get(v,v)

def _cos2(Xc,yc,vn=5):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lda=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"); lda.fit(Xc,yc)
        scal=lda.scalings_ if hasattr(lda,"scalings_") else lda.coef_.T
        out=np.full((3,vn),np.nan)
        for k in range(min(3,scal.shape[1])):
            v=scal[:,k]; nn=np.linalg.norm(v)
            if nn>=1e-12: out[k]=(v/nn)**2
        return out
    except Exception:
        return np.full((3,vn),np.nan)
def _cos2_grid(X,y,n_min,vn=5,seed=0):
    rng=np.random.default_rng(seed); grid=np.full((X.shape[1],3,vn),np.nan); cls=np.unique(y)
    for ti in range(X.shape[1]):
        idx=[]; ok=True
        for c in cls:
            ci=np.where(y==c)[0]
            if len(ci)<n_min: ok=False; break
            idx.extend(rng.choice(ci,n_min,replace=False).tolist())
        if not ok: continue
        idx=np.array(idx); Xb=X[idx,ti,:]; yb=y[idx]; keep=~np.any(np.isnan(Xb),axis=1)
        if keep.sum()<4 or len(np.unique(yb[keep]))<2: continue
        grid[ti]=_cos2(Xb[keep],yb[keep],vn)
    return grid

# ---- subset / balancing builders: each returns (X_sub, y01, strata, n_min, chance) ----
def _design(ev, which):
    """Return list of (label, X, y, n_min, chance) for the requested design."""
    X=ev["X"]; yw=ev["yw"]; yo=ev["yo"]
    base=(yw=="BaseCondition"); full=(yw=="FullLoopAR")
    succ=(yo==1); fail=(yo==0)
    rng=np.random.default_rng(_BAL_SEED)
    def nmin(y):
        cl=np.unique(y); return int(min((y==c).sum() for c in cl)) if len(cl)>=2 else 0

    if which=="BvF_unbal":
        m=base|full; y=np.where(yw[m]=="BaseCondition",0,1)
        return ("Base vs Full (unbalanced)", X[m], y, nmin(y), 0.5)

    if which=="BvF_bal":
        # equal succ/fail per arm: within each outcome, take min(base,full); pool
        selB=[]; selF=[]
        for out_mask in [succ,fail]:
            ib=np.where(base&out_mask)[0]; iff=np.where(full&out_mask)[0]
            k=min(len(ib),len(iff))
            if k==0: continue
            selB+=rng.choice(ib,k,replace=False).tolist(); selF+=rng.choice(iff,k,replace=False).tolist()
        if not selB: return None
        idx=np.array(selB+selF); y=np.array([0]*len(selB)+[1]*len(selF))
        return ("Base vs Full (balanced by outcome)", X[idx], y, nmin(y), 0.5)

    if which=="SvF_unbal":
        m=succ|fail; y=yo[m].astype(int)
        return ("Success vs Failure (unbalanced)", X[m], y, nmin(y), 0.5)

    if which=="SvF_bal":
        # equal condition mix per arm: within each of 4 conditions take min(succ,fail)
        selS=[]; selF=[]
        for cc in np.unique(yw):
            isc=np.where((yw==cc)&succ)[0]; ifc=np.where((yw==cc)&fail)[0]
            k=min(len(isc),len(ifc))
            if k==0: continue
            selS+=rng.choice(isc,k,replace=False).tolist(); selF+=rng.choice(ifc,k,replace=False).tolist()
        if not selS: return None
        idx=np.array(selS+selF); y=np.array([1]*len(selS)+[0]*len(selF))
        return ("Success vs Failure (balanced by condition)", X[idx], y, nmin(y), 0.5)

    if which=="SvF_base":
        m=base&(succ|fail); y=yo[m].astype(int)
        return ("Success vs Failure (Base only)", X[m], y, nmin(y), 0.5)

    if which=="SvF_full":
        m=full&(succ|fail); y=yo[m].astype(int)
        return ("Success vs Failure (Full only)", X[m], y, nmin(y), 0.5)
    raise ValueError(which)

_DESIGNS = ["BvF_unbal","BvF_bal","SvF_unbal","SvF_bal","SvF_base","SvF_full"]
_DCOL = {"BvF_unbal":"#9467bd","BvF_bal":"#6a3d9a","SvF_unbal":"#d62728","SvF_bal":"#a01619",
         "SvF_base":"#1f77b4","SvF_full":"#ff7f0e"}

def _run(label, X, y, n_min, ch, col):
    if n_min<3:
        print(f"    {label}: n_min/class={n_min} <3, skip"); return None
    acc,nt=_acc_tc(X,y,n_min)
    tfce_obs=_tfce_1d(np.nan_to_num(_z(acc,ch,nt),nan=0.0))
    rng=np.random.default_rng(7); null=np.empty(_TFCE_B)
    for b in _tqdm(range(_TFCE_B),desc=label[:18],leave=False):
        accp,np_=_acc_tc(X,rng.permutation(y),n_min,seed=b+1)
        null[b]=_tfce_1d(np.nan_to_num(_z(accp,ch,np_),nan=0.0)).max()
    p=_pval(tfce_obs,null); sig=p<_TFCE_ALPHA
    return dict(acc=acc,p=p,sig=sig,clusters=_clusters(sig),ch=ch,col=col,n=nt,
                n_min=n_min,cos2_grid=_cos2_grid(X,y,n_min),label=label)

def _plot(results, ts, title):
    items=[r for r in results if r is not None]
    if not items: print(f"  {title}: nothing to plot"); return
    nC=len(items); fig,axes=plt.subplots(2,nC,figsize=(4.6*nC,8))
    if nC==1: axes=axes.reshape(2,1)
    fig.suptitle(f"{title}: TFCE-LDA car frame (B={_TFCE_B})",fontsize=12,fontweight="bold")
    for ci,info in enumerate(items):
        ax=axes[0,ci]
        ax.plot(ts,_cf_smooth(info["acc"]),color=info["col"],lw=1.6)
        ax.axhline(info["ch"],color="gray",lw=0.6,ls=":"); ax.axvline(0,color="k",lw=0.8,ls="--")
        for lo,hi in (info["clusters"] or []):
            ax.axvspan(ts[lo],ts[hi-1] if hi-1<len(ts) else ts[-1],color=info["col"],alpha=0.20)
        ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
        ax.set_title(f"{info['label']}\nn_min/class={info['n_min']}  clusters={len(info['clusters'] or [])}",fontsize=8)
        ax=axes[1,ci]; cl=info["clusters"] or []
        if not cl:
            ax.text(0.5,0.5,"no sig clusters",ha="center",va="center",transform=ax.transAxes); ax.axis("off"); continue
        bw=0.8/max(len(cl),1); x=np.arange(len(_CF_VARS))
        for ki,(lo,hi) in enumerate(cl):
            ax.bar(x+ki*bw,np.nanmean(info["cos2_grid"][lo:hi],axis=(0,1)),bw,
                   label=f"[{ts[lo]:.2f},{ts[hi-1] if hi-1<len(ts) else ts[-1]:.2f}]")
        ax.set_xticks(x+bw*(len(cl)-1)/2)
        ax.set_xticklabels([_vlab(v) for v in _CF_VARS],rotation=20,ha="right",fontsize=8)
        ax.set_ylabel("cos2 (avg LD1-3)"); ax.set_ylim(0,1.05); ax.legend(fontsize=7)
    plt.tight_layout(rect=[0,0,1,0.95]); plt.close("all")

# ---- load ----
print("Loading car-frame events...")
_ev={}
for _e in _CF_EVENTS:
    _df=_cf_load(_e)
    if _df is None: continue
    ts,X,yw,yo=_build(_df,_CF_VARS); _ev[_e]=dict(ts=ts,X=X,yw=yw,yo=yo)
    print(f"  {_e}: {X.shape[0]} drivers, {len(ts)} timepoints")

# ---- run per event ----
_lda_results={}
for _e,_d in _ev.items():
    print(f"=== {_e} ==="); _lda_results[_e]=[]
    for which in _DESIGNS:
        des=_design(_d,which)
        if des is None: _lda_results[_e].append(None); continue
        lbl,Xs,ys,nm,ch=des
        print(f"  {which}: n={Xs.shape[0]} n_min/class={nm}")
        _lda_results[_e].append(_run(lbl,Xs,ys,nm,ch,_DCOL[which]))
    _plot(_lda_results[_e], _d["ts"], _e)

# ---- compact summary ----
print("\\nSIG CLUSTERS (p<0.05):")
for _e,res in _lda_results.items():
    for info in res:
        if info is None: continue
        ts=_ev[_e]["ts"]; cl=info["clusters"] or []
        spans=", ".join(f"[{ts[lo]:.2f},{ts[hi-1] if hi-1<len(ts) else ts[-1]:.2f}]" for lo,hi in cl) or "none"
        print(f"  {_e:<22}{info['label']:<40}n_min={info['n_min']:<3}{spans}")
print("\\nNOTE: Base-only / Full-only outcome tests have small n_min; a null there")
print("may be underpowering, not absence of an outcome effect.")
