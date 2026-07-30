import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
already=any("TFCE ON 1-DIM (PC1) LDA" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "cell present."
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "LDA ON PCA COMPONENTS" in "".join(c["source"]): ins=i
assert ins is not None

code = r'''# ============================================================
# TFCE ON 1-DIM (PC1) LDA  -- significance of decoding from PC1 alone
# ------------------------------------------------------------
# Per timepoint: gold PCA (C + pooled), take PC1 score only, balanced-subsample
# LDA CV accuracy. TFCE on z=(acc-chance)/SE; B label-permutations -> null TFCE
# max -> sig clusters. Shows ONLY the 1-dim accuracy curve with sig clusters
# overlaid (green x-axis). Same balanced contrasts. Reuses _ev, _design, and the
# TFCE helpers (_tfce_1d,_z,_pval,_clusters,_xaxis_clusters,_cf_smooth) + the
# PCA helpers (_pd_gold_prep,_pd_pca_feats) from the cells above.
# ============================================================
import numpy as np, warnings
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

_T1_DESIGNS=["BvF_bal","SvF_bal","SvF_base","SvF_full"]
_T1_COL="#1b9e77"

def _t1_acc(Xg, y, n_min, rng):
    """1-dim (PC1) LDA CV accuracy over time on already-gold-prepped Xg."""
    T=Xg.shape[1]; acc=np.full(T,np.nan); cls=np.unique(y)
    for t in range(T):
        sc,_=_pd_pca_feats(Xg[:,t,:]); feat=sc[:,:1]
        idx=[]; ok=True
        for cc in cls:
            ci=np.where(y==cc)[0]
            if len(ci)<n_min: ok=False; break
            idx.extend(rng.choice(ci,n_min,replace=False).tolist())
        if not ok: continue
        idx=np.array(idx); Xb=feat[idx]; yb=y[idx]
        keep=~np.any(np.isnan(Xb),axis=1); Xb=Xb[keep]; yb=yb[keep]
        if len(np.unique(yb))<2 or len(Xb)<6: continue
        cor=tot=0
        for tr,te in StratifiedKFold(min(5,len(Xb)//2),shuffle=True,random_state=0).split(Xb,yb):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto").fit(Xb[tr],yb[tr])
                cor+=int((m.predict(Xb[te])==yb[te]).sum()); tot+=len(te)
        if tot: acc[t]=cor/tot
    return acc

def _t1_run(Xs, ys, nm, ch):
    Xg=_pd_gold_prep(Xs)
    rng=np.random.default_rng(0)
    acc=_t1_acc(Xg,ys,nm,rng)
    # n for z: balanced total = 2*nm (per timepoint test size)
    nz=2*nm
    tfce_obs=_tfce_1d(np.nan_to_num(_z(acc,ch,nz),nan=0.0))
    rng2=np.random.default_rng(7); null=np.empty(_TFCE_B)
    for b in range(_TFCE_B):
        accp=_t1_acc(Xg,rng2.permutation(ys),nm,np.random.default_rng(b+1))
        null[b]=_tfce_1d(np.nan_to_num(_z(accp,ch,nz),nan=0.0)).max()
    p=_pval(tfce_obs,null); sig=p<_TFCE_ALPHA
    return acc,_clusters(sig)

print("TFCE on 1-dim (PC1) LDA -- per event, sig clusters overlaid")
for _e in _ev:
    ts=_ev[_e]["ts"]
    for which in _T1_DESIGNS:
        des=_design(_ev[_e],which)
        if des is None: continue
        lbl,Xs,ys,nm,ch=des
        if nm<3: print(f"{_e} | {lbl}: low n"); continue
        print(f"  {_e} | {lbl}: B={_TFCE_B} perms")
        acc,clusters=_t1_run(Xs,ys,nm,ch)
        fig,ax=plt.subplots(figsize=(9,4))
        ax.plot(ts,_cf_smooth(acc),color=_T1_COL,lw=1.6,label="1-dim (PC1) accuracy")
        ax.axhline(ch,color="gray",lw=0.8,ls=":"); ax.axvline(0,color="k",lw=1.0,ls="--")
        _xaxis_clusters(ax, ts, clusters)
        ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
        ax.set_title(f"{_e} -- {lbl}  (1-dim LDA; green = TFCE p<{_TFCE_ALPHA})",fontsize=11)
        ax.grid(True,alpha=0.25); ax.legend(fontsize=8,loc="lower right")
        plt.tight_layout(); plt.show()
print("Done. 1-dim = decoding from PC1 (top variance) only; green x-axis = TFCE-significant.")
'''
ast.parse(code)
nb["cells"].insert(ins+1, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted TFCE 1-dim cell after {ins}. Total: {len(nb['cells'])}.")
