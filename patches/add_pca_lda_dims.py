import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
already=any("LDA ON PCA COMPONENTS (1 / 2 / 5 dims)" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "cell present."

ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "one fused figure per contrast" in "".join(c["source"]): ins=i
if ins is None:
    for i,c in enumerate(nb["cells"]):
        if c["cell_type"]=="code" and "TFCE-LDA" in "".join(c["source"]) and "def _run" in "".join(c["source"]): ins=i
assert ins is not None

code = r'''# ============================================================
# LDA ON PCA COMPONENTS (1 / 2 / 5 dims)  -- no TFCE, CV accuracy only
# ------------------------------------------------------------
# Per timepoint: gold-standardise the 5 car-frame channels across drivers, PCA
# (unsupervised, fit on all drivers), then feed LDA the top PCs:
#   1-dim = PC1 ;  2-dim = PC1+PC2 ;  5-dim = all PCs (full).
# "Top" = PCA explained-variance order. Balanced-subsample 5-fold CV accuracy.
# Same balanced contrasts. No permutation test, no gradient shading.
# Heatmap (top-LDA = 1-dim): PC1 loadings (cos^2) per channel over time.
# Reuses _ev (raw X) + _design (labels/balancing) from the TFCE-LDA cell.
# ============================================================
import numpy as np, warnings
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from mpl_toolkits.axes_grid1 import make_axes_locatable

assert "_ev" in globals() and "_design" in globals(), "Run the TFCE-LDA cell first."

_PD_VLAB=["Head.x","Head.y","Eye.x","Eye.y","Steering"]
_PD_M=5
_PD_DIMS=[1,2,5]
_PD_DCOL={1:"#1b9e77",2:"#7570b3",5:"#d95f02"}
_PD_DESIGNS=["BvF_bal","SvF_bal","SvF_base","SvF_full"]

def _pd_pca_feats(Xt):
    """Xt: (W, M) at one timepoint. Gold-standardise across drivers, PCA.
    Returns scores (W, M) in explained-variance order + PC1 cos^2 (M,)."""
    A=Xt.astype(float)
    A=A-A.mean(0)                                   # cross-driver center at t
    sd=A.std(0,ddof=1); sd[sd<1e-12]=1.0; A=A/sd    # pooled per-channel scale (per t)
    cov=np.cov(A.T)
    ev,evec=np.linalg.eigh(cov)
    o=np.argsort(ev)[::-1]; evec=evec[:,o]
    scores=A@evec                                   # (W, M) PC scores, PV order
    pc1cos2=evec[:,0]**2
    return scores, pc1cos2

def _pd_acc_curve(X, y, n_min, k_dims):
    """LDA CV accuracy over time using top-k_dims PCA scores. y has labels (0/1)."""
    T=X.shape[1]; acc=np.full(T,np.nan); pc1grid=np.full((_PD_M,T),np.nan)
    rng=np.random.default_rng(0); cls=np.unique(y)
    for t in range(T):
        sc,pc1=_pd_pca_feats(X[:,t,:]); pc1grid[:,t]=pc1
        feat=sc[:,:k_dims]
        idx=[]
        ok=True
        for ccls in cls:
            ci=np.where(y==ccls)[0]
            if len(ci)<n_min: ok=False; break
            idx.extend(rng.choice(ci,n_min,replace=False).tolist())
        if not ok: continue
        idx=np.array(idx); Xb=feat[idx]; yb=y[idx]
        keep=~np.any(np.isnan(Xb),axis=1)
        Xb=Xb[keep]; yb=yb[keep]
        if len(np.unique(yb))<2 or len(Xb)<6: continue
        cor=tot=0
        for tr,te in StratifiedKFold(min(5,len(Xb)//2),shuffle=True,random_state=0).split(Xb,yb):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto").fit(Xb[tr],yb[tr])
                cor+=int((m.predict(Xb[te])==yb[te]).sum()); tot+=len(te)
        if tot: acc[t]=cor/tot
    return acc, pc1grid

def _pd_smooth(x,k=5):
    x=np.asarray(x,float); out=x.copy()
    for i in range(len(x)):
        seg=x[max(0,i-k//2):min(len(x),i+k//2+1)]; seg=seg[~np.isnan(seg)]
        if len(seg): out[i]=seg.mean()
    return out

# ---- per event, per contrast: one fused figure (accuracy 1/2/5 + PC1 heatmap) ----
for _e in _ev:
    ts=_ev[_e]["ts"]
    for which in _PD_DESIGNS:
        des=_design(_ev[_e],which)
        if des is None: continue
        lbl,Xs,ys,nm,ch=des
        if nm<3: print(f"{_e} | {lbl}: low n"); continue
        accs={}; pc1grid=None
        for kd in _PD_DIMS:
            a,g=_pd_acc_curve(Xs,ys,nm,kd); accs[kd]=a
            if kd==1: pc1grid=g
        fig,(axA,axH)=plt.subplots(2,1,figsize=(9,5.6),sharex=True,
                                   gridspec_kw={"height_ratios":[2.3,1.0],"hspace":0.08})
        for kd in _PD_DIMS:
            axA.plot(ts,_pd_smooth(accs[kd]),color=_PD_DCOL[kd],lw=2,label=f"{kd}-dim (top {kd} PC)")
        axA.axhline(ch,color="gray",lw=0.8,ls=":"); axA.axvline(0,color="k",lw=1.0,ls="--")
        axA.set_ylim(0,1); axA.set_ylabel("CV accuracy"); axA.legend(fontsize=8,loc="lower right")
        axA.set_title(f"{_e} -- {lbl}   (LDA on top PCA dims)",fontsize=11); axA.grid(True,alpha=0.25)
        _divA=make_axes_locatable(axA); _cA=_divA.append_axes("right",size="2.5%",pad=0.08); _cA.axis("off")
        # heatmap: PC1 loadings (cos^2) over time -- the dimension the 1-dim LDA used
        im=axH.imshow(pc1grid,aspect="auto",origin="lower",extent=[ts[0],ts[-1],-0.5,_PD_M-0.5],
                      cmap="magma",vmin=0,vmax=1)
        axH.axvline(0,color="white",ls="--",lw=0.8,alpha=0.7)
        axH.set_yticks(range(_PD_M)); axH.set_yticklabels(_PD_VLAB,fontsize=8)
        axH.set_xlabel("Time (s)"); axH.set_ylabel("PC1 cos²",fontsize=9)
        _divH=make_axes_locatable(axH); _cH=_divH.append_axes("right",size="2.5%",pad=0.08)
        fig.colorbar(im,cax=_cH,label="cos²")
        plt.tight_layout(); plt.show()
print("Accuracy: LDA on top-1/2/5 PCA components per timepoint (gold standardised, no TFCE).")
print("Heatmap: PC1 channel loadings (cos²) -- the single dimension the 1-dim LDA decodes from.")
'''
ast.parse(code)
nb["cells"].insert(ins+1, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted PCA-dims LDA cell after {ins}. Total: {len(nb['cells'])}.")
