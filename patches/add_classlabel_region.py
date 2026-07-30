import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
already=any("1-DIM LDA: CLASS-LABEL SHUFFLE REGION TEST" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "cell present."
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "TFCE ON 1-DIM (PC1) LDA" in "".join(c["source"]): ins=i
assert ins is not None

code = r'''# ============================================================
# 1-DIM LDA: CLASS-LABEL SHUFFLE REGION TEST  (rigorous null)
# ------------------------------------------------------------
# Stronger than the region-label shuffle: the NULL shuffles the CLASS labels
# (Success/Failure or Base/Full) and recomputes the whole 1-dim PC1 LDA accuracy
# curve, then takes its region means. So the p answers "is the pre->post (or
# TOR->Action) change bigger than what CHANCE decoding produces", controlling
# for both autocorrelation and the chance baseline.
#   Contrast 1: Post - Pre.
#   Contrast 2: Baseline/TOR/Action pairwise (Base-TOR, TOR-Action) + omnibus.
# Re-fits LDA inside the null (slow). Per event + average. Reuses _ev, _design,
# _pd_gold_prep, _pd_pca_feats.
# ============================================================
import numpy as np, os, warnings
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

_CL_PERM=2000        # class-label permutations (raise for final)
_CL_ROBS=20          # balanced subsample draws for the OBSERVED curve (stabilise)
_CL_DESIGNS=["BvF_bal","SvF_bal","SvF_base","SvF_full"]

_clrt=pd.read_csv(os.getcwd()+"/reaction_times.csv")
_clrt["RT_Steering"]=pd.to_numeric(_clrt["RT_Steering"],errors="coerce")
_clrt=_clrt[(_clrt["Unavailable_Steering"].astype(str).str.lower()!="true") & _clrt["RT_Steering"].notna()]
def _cl_mrt(e):
    s=_clrt[_clrt["EventName"]==e]; return float(s["RT_Steering"].mean()) if len(s) else np.nan

def _cl_acc(Xg, y, nm, rng):
    """1-dim PC1 LDA accuracy curve on already-gold-prepped Xg, one balanced draw."""
    T=Xg.shape[1]; acc=np.full(T,np.nan); cls=np.unique(y)
    for t in range(T):
        sc,_=_pd_pca_feats(Xg[:,t,:]); feat=sc[:,:1]
        idx=[]; ok=True
        for cc in cls:
            ci=np.where(y==cc)[0]
            if len(ci)<nm: ok=False; break
            idx.extend(rng.choice(ci,nm,replace=False).tolist())
        if not ok: continue
        idx=np.array(idx); Xb=feat[idx]; yb=y[idx]; k=~np.any(np.isnan(Xb),1); Xb=Xb[k]; yb=yb[k]
        if len(np.unique(yb))<2 or len(Xb)<6: continue
        cor=tot=0
        for tr,te in StratifiedKFold(min(5,len(Xb)//2),shuffle=True,random_state=0).split(Xb,yb):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto").fit(Xb[tr],yb[tr])
                cor+=int((m.predict(Xb[te])==yb[te]).sum()); tot+=len(te)
        if tot: acc[t]=cor/tot
    return acc

def _cl_regs(secs,rt):
    return dict(Pre=(secs>=-4)&(secs<0),Post=(secs>=0)&(secs<=4),
               Baseline=(secs>=-4)&(secs<0),TOR=(secs>=0)&(secs<rt),Action=(secs>=rt)&(secs<=4))

def _cl_stats(acc,R):
    pp=np.nanmean(acc[R['Post']])-np.nanmean(acc[R['Pre']])
    bt=np.nanmean(acc[R['Baseline']])-np.nanmean(acc[R['TOR']])
    ta=np.nanmean(acc[R['TOR']])-np.nanmean(acc[R['Action']])
    ba=np.nanmean(acc[R['Baseline']])-np.nanmean(acc[R['Action']])
    omni=np.var([np.nanmean(acc[R['Baseline']]),np.nanmean(acc[R['TOR']]),np.nanmean(acc[R['Action']])])
    return dict(PostPre=pp, BaseTOR=bt, TORAct=ta, BaseAct=ba, omni=omni)

def _cl_run(Xs, ys, nm, rt, ts):
    Xg=_pd_gold_prep(Xs); R=_cl_regs(ts,rt)
    # observed: mean over R balanced draws
    obs_acc=np.nanmean([_cl_acc(Xg,ys,nm,np.random.default_rng(r)) for r in range(_CL_ROBS)],0)
    obs=_cl_stats(obs_acc,R)
    # null: shuffle class labels, recompute one curve per perm
    rng=np.random.default_rng(7)
    keys=["PostPre","BaseTOR","TORAct","BaseAct","omni"]
    null={k:np.empty(_CL_PERM) for k in keys}
    for p in range(_CL_PERM):
        accp=_cl_acc(Xg, rng.permutation(ys), nm, np.random.default_rng(p+1))
        st=_cl_stats(accp,R)
        for k in keys: null[k][p]=st[k]
    pv={}
    for k in keys:
        n=null[k]; n=n[~np.isnan(n)]; o=obs[k]
        if k=="omni":  pv[k]=float((np.sum(n>=o)+1)/(len(n)+1))            # one-tailed
        else:          pv[k]=float((np.sum(np.abs(n)>=abs(o))+1)/(len(n)+1)) # two-sided
    return obs, pv

def _st(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""

print("="*94)
print("1-DIM LDA CLASS-LABEL SHUFFLE TEST  (null = shuffle class labels, recompute curve)")
print("="*94)
print(f"{'contrast':<22}{'event':<14}{'Post-Pre(p)':>16}{'TOR-Act(p)':>15}{'Base-TOR(p)':>15}{'omni p':>9}")
print("-"*94)
_cl_res={w:{} for w in _CL_DESIGNS}
for which in _CL_DESIGNS:
    for _e in _ev:
        des=_design(_ev[_e],which)
        if des is None: continue
        lbl,Xs,ys,nm,ch=des
        if nm<3: print(f"{which:<22}{_e:<14} low n"); continue
        obs,pv=_cl_run(Xs,ys,nm,_cl_mrt(_e),_ev[_e]["ts"])
        _cl_res[which][_e]=(obs,pv)
        print(f"{which:<22}{_e:<14}{obs['PostPre']:>+9.3f}({pv['PostPre']:.3f}){_st(pv['PostPre']):<3}"
              f"{obs['TORAct']:>+8.3f}({pv['TORAct']:.3f}){_st(pv['TORAct']):<3}"
              f"{obs['BaseTOR']:>+8.3f}({pv['BaseTOR']:.3f}){_st(pv['BaseTOR']):<3}{pv['omni']:>9.3f}{_st(pv['omni'])}")
    print("-"*94)
print(f"\\nClass-label shuffle, {_CL_PERM} perms; observed averaged over {_CL_ROBS} balanced draws.")
print("Post-Pre/TOR-Act/Base-TOR two-sided; omni one-tailed. * p<.05 ** p<.01 *** p<.001.")
print("Note: slower than region-shuffle (re-fits LDA per perm) but controls chance baseline + autocorrelation.")
'''
ast.parse(code)
nb["cells"].insert(ins, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted class-label region test cell at {ins} (before TFCE). Total: {len(nb['cells'])}.")
