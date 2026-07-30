import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
already=any("1-DIM (PC1) LDA: PRE/POST + 3-REGION" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "cell present."
# insert just before the TFCE 1-dim cell
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "TFCE ON 1-DIM (PC1) LDA" in "".join(c["source"]): ins=i
assert ins is not None

code = r'''# ============================================================
# 1-DIM (PC1) LDA: PRE/POST + 3-REGION (Baseline/TOR/Action) accuracy stats
# ------------------------------------------------------------
# 1-dim accuracy = gold PCA -> PC1 score -> balanced LDA CV accuracy over time.
# Region-label shuffle permutation (over timepoints, sizes fixed):
#   Contrast 1: Pre [-4,0) vs Post [0,4]  (two-sided).
#   Contrast 2: Baseline [-4,0) | TOR [0,meanRT) | Action [meanRT,4]; pairwise
#               diffs + omnibus. Boundary = mean RT_Steering per event.
# Per event + average across events. No LDA re-fit in the null (shuffles the
# computed accuracy curve), so this is fast. Reuses _ev, _design, _pd_gold_prep,
# _pd_pca_feats.
# ============================================================
import numpy as np, os, warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

_R1_PERM=5000; _R1_RNG=np.random.default_rng(0)
_R1_DESIGNS=["BvF_bal","SvF_bal","SvF_base","SvF_full"]

# mean RT per event
_r1rt=pd.read_csv(os.getcwd()+"/reaction_times.csv")
_r1rt["RT_Steering"]=pd.to_numeric(_r1rt["RT_Steering"],errors="coerce")
_r1rt=_r1rt[(_r1rt["Unavailable_Steering"].astype(str).str.lower()!="true") & _r1rt["RT_Steering"].notna()]
def _r1_mrt(e):
    s=_r1rt[_r1rt["EventName"]==e]; return float(s["RT_Steering"].mean()) if len(s) else np.nan

def _r1_acc(Xs,y,nm):
    """1-dim PC1 LDA CV accuracy over time (gold PCA)."""
    Xg=_pd_gold_prep(Xs); T=Xg.shape[1]; acc=np.full(T,np.nan); cls=np.unique(y); rng=np.random.default_rng(0)
    for t in range(T):
        sc,_=_pd_pca_feats(Xg[:,t,:]); feat=sc[:,:1]
        idx=[]; ok=True
        for cc in cls:
            ci=np.where(y==cc)[0]
            if len(ci)<nm: ok=False; break
            idx.extend(rng.choice(ci,nm,replace=False).tolist())
        if not ok: continue
        idx=np.array(idx); Xb=feat[idx]; yb=y[idx]; keep=~np.any(np.isnan(Xb),1); Xb=Xb[keep]; yb=yb[keep]
        if len(np.unique(yb))<2 or len(Xb)<6: continue
        cor=tot=0
        for tr,te in StratifiedKFold(min(5,len(Xb)//2),shuffle=True,random_state=0).split(Xb,yb):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto").fit(Xb[tr],yb[tr])
                cor+=int((m.predict(Xb[te])==yb[te]).sum()); tot+=len(te)
        if tot: acc[t]=cor/tot
    return acc

def _regs(secs,rt):
    return dict(Pre=(secs>=-4)&(secs<0),Post=(secs>=0)&(secs<=4),
               Baseline=(secs>=-4)&(secs<0),TOR=(secs>=0)&(secs<rt),Action=(secs>=rt)&(secs<=4))
def _p2(c,mA,mB,P=_R1_PERM,rng=None):
    rng=rng or _R1_RNG; idx=np.where(mA|mB)[0]; nA=int(mA.sum()); obs=np.nanmean(c[mB])-np.nanmean(c[mA])
    v=c[idx]; v=v[~np.isnan(v)]; null=np.empty(P); a=np.arange(len(v))
    for p in range(P): rng.shuffle(a); null[p]=v[a[:len(v)-nA]].mean()-v[a[len(v)-nA:]].mean()
    return obs,float((np.sum(np.abs(null)>=abs(obs))+1)/(P+1))
def _p3(c,mB,mT,mA,P=_R1_PERM,rng=None):
    rng=rng or _R1_RNG; idx=np.where(mB|mT|mA)[0]; v=c[idx]; m=~np.isnan(v); v=v[m]
    nB=int(np.sum(~np.isnan(c[mB]))); nT=int(np.sum(~np.isnan(c[mT])))
    mb,mt,ma=np.nanmean(c[mB]),np.nanmean(c[mT]),np.nanmean(c[mA])
    op={"Base-TOR":mb-mt,"TOR-Act":mt-ma,"Base-Act":mb-ma}; oo=np.var([mb,mt,ma])
    npp={k:np.empty(P) for k in op}; no=np.empty(P); a=np.arange(len(v))
    for p in range(P):
        rng.shuffle(a); b=v[a[:nB]].mean(); t=v[a[nB:nB+nT]].mean(); ac=v[a[nB+nT:]].mean()
        npp["Base-TOR"][p]=b-t; npp["TOR-Act"][p]=t-ac; npp["Base-Act"][p]=b-ac; no[p]=np.var([b,t,ac])
    pv={k:float((np.sum(np.abs(npp[k])>=abs(op[k]))+1)/(P+1)) for k in op}
    return op,pv,float((np.sum(no>=oo)+1)/(P+1))
def _st(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""

# compute 1-dim accuracy per (contrast,event); store for averaging
_R1={w:{} for w in _R1_DESIGNS}; _R1RT={}
for _e in _ev:
    for which in _R1_DESIGNS:
        des=_design(_ev[_e],which)
        if des is None: continue
        lbl,Xs,ys,nm,ch=des
        if nm<3: continue
        _R1[which][_e]=(_ev[_e]["ts"],_r1_acc(Xs,ys,nm)); _R1RT[_e]=_r1_mrt(_e)

print("="*82); print("1-DIM (PC1) LDA accuracy -- CONTRAST 1: Pre vs Post (region-label shuffle)"); print("="*82)
print(f"{'contrast':<22}{'event':<14}{'pre':>7}{'post':>7}{'post-pre':>9}{'p':>8}"); print("-"*82)
for which in _R1_DESIGNS:
    for _e in _ev:
        if _e not in _R1[which]: continue
        ts,acc=_R1[which][_e]; R=_regs(ts,_R1RT[_e]); o,p=_p2(acc,R['Pre'],R['Post'])
        print(f"{which:<22}{_e:<14}{np.nanmean(acc[R['Pre']]):>7.3f}{np.nanmean(acc[R['Post']]):>7.3f}{o:>+9.3f}{p:>8.3f}{_st(p)}")
    # average across events
    ev_e=[e for e in _ev if e in _R1[which]]
    if len(ev_e)>1:
        ref=_R1[which][ev_e[0]][0]
        def _al(t,c): return c if (len(t)==len(ref) and np.allclose(t,ref)) else np.interp(ref,t,c,left=np.nan,right=np.nan)
        avg=np.nanmean([_al(_R1[which][e][0],_R1[which][e][1]) for e in ev_e],0)
        R=_regs(ref,np.nanmean([_R1RT[e] for e in ev_e])); o,p=_p2(avg,R['Pre'],R['Post'])
        print(f"{which:<22}{'AVERAGE':<14}{np.nanmean(avg[R['Pre']]):>7.3f}{np.nanmean(avg[R['Post']]):>7.3f}{o:>+9.3f}{p:>8.3f}{_st(p)}")
    print("-"*82)

print("\n"+"="*92); print("CONTRAST 2: Baseline / TOR / Action (pairwise + omnibus), 1-dim PC1 LDA accuracy"); print("="*92)
print(f"{'contrast':<22}{'event':<14}{'Base':>6}{'TOR':>6}{'Act':>6} | {'B-TOR(p)':>13}{'TOR-A(p)':>13}{'B-A(p)':>13}{'omni':>7}"); print("-"*98)
for which in _R1_DESIGNS:
    for _e in _ev:
        if _e not in _R1[which]: continue
        ts,acc=_R1[which][_e]; rt=_R1RT[_e]; R=_regs(ts,rt)
        if R['TOR'].sum()<2 or R['Action'].sum()<2: print(f"{which:<22}{_e:<14} TOR/Act short"); continue
        mb,mt,ma=np.nanmean(acc[R['Baseline']]),np.nanmean(acc[R['TOR']]),np.nanmean(acc[R['Action']])
        op,pv,po=_p3(acc,R['Baseline'],R['TOR'],R['Action'])
        print(f"{which:<22}{_e:<14}{mb:>6.2f}{mt:>6.2f}{ma:>6.2f} | "
              f"{op['Base-TOR']:>+6.2f}({pv['Base-TOR']:.3f}){_st(pv['Base-TOR']):<2}"
              f"{op['TOR-Act']:>+6.2f}({pv['TOR-Act']:.3f}){_st(pv['TOR-Act']):<2}"
              f"{op['Base-Act']:>+6.2f}({pv['Base-Act']:.3f}){_st(pv['Base-Act']):<2}{po:>7.3f}{_st(po)}")
    ev_e=[e for e in _ev if e in _R1[which]]
    if len(ev_e)>1:
        ref=_R1[which][ev_e[0]][0]
        def _al2(t,c): return c if (len(t)==len(ref) and np.allclose(t,ref)) else np.interp(ref,t,c,left=np.nan,right=np.nan)
        avg=np.nanmean([_al2(_R1[which][e][0],_R1[which][e][1]) for e in ev_e],0)
        rt=np.nanmean([_R1RT[e] for e in ev_e]); R=_regs(ref,rt)
        mb,mt,ma=np.nanmean(avg[R['Baseline']]),np.nanmean(avg[R['TOR']]),np.nanmean(avg[R['Action']])
        op,pv,po=_p3(avg,R['Baseline'],R['TOR'],R['Action'])
        print(f"{which:<22}{'AVERAGE':<14}{mb:>6.2f}{mt:>6.2f}{ma:>6.2f} | "
              f"{op['Base-TOR']:>+6.2f}({pv['Base-TOR']:.3f}){_st(pv['Base-TOR']):<2}"
              f"{op['TOR-Act']:>+6.2f}({pv['TOR-Act']:.3f}){_st(pv['TOR-Act']):<2}"
              f"{op['Base-Act']:>+6.2f}({pv['Base-Act']:.3f}){_st(pv['Base-Act']):<2}{po:>7.3f}{_st(po)}")
    print("-"*98)
print("\n1-dim = PC1-only LDA. Region-label shuffle ("+str(_R1_PERM)+" perms). * p<.05 ** p<.01 *** p<.001.")
'''
ast.parse(code)
nb["cells"].insert(ins, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted 1-dim pre/post + 3-region stats cell at {ins} (before TFCE). Total: {len(nb['cells'])}.")
