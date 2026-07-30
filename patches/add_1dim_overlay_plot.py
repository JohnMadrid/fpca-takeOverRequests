import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
already=any("1-DIM LDA: REGION MEANS OVERLAID ON CURVES" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "cell present."
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "TFCE ON 1-DIM (PC1) LDA" in "".join(c["source"]): ins=i
assert ins is not None

code = r'''# ============================================================
# 1-DIM LDA: REGION MEANS OVERLAID ON CURVES (pre/post + Baseline/TOR/Action)
# ------------------------------------------------------------
# Per contrast: stacked panels (3 events + average). Each panel = the 1-dim
# (PC1) LDA accuracy time-course, with horizontal region-mean segments drawn on
# top of the curve over each region's time window:
#   pre/post (gray)  and  Baseline/TOR/Action (coloured).
# Stars from per-event region-label shuffle p (TOR vs Baseline, Action vs TOR).
# Reuses _R1 (acc curves) + _R1RT (mean RT) from the stats cell above; if absent,
# recomputes via _ev/_design/_pd_gold_prep/_pd_pca_feats.
# ============================================================
import numpy as np, os, warnings
import matplotlib.pyplot as plt

_OV_DESIGNS=["BvF_bal","SvF_bal","SvF_base","SvF_full"]
_OV_PERM=5000; _OV_RNG=np.random.default_rng(0)

# reuse _R1/_R1RT if the stats cell ran; else recompute
if "_R1" not in globals() or "_R1RT" not in globals():
    import pandas as pd
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    _rt=pd.read_csv(os.getcwd()+"/reaction_times.csv"); _rt["RT_Steering"]=pd.to_numeric(_rt["RT_Steering"],errors="coerce")
    _rt=_rt[(_rt["Unavailable_Steering"].astype(str).str.lower()!="true") & _rt["RT_Steering"].notna()]
    def _mrt(e):
        s=_rt[_rt["EventName"]==e]; return float(s["RT_Steering"].mean()) if len(s) else np.nan
    def _acc1(Xs,y,nm):
        Xg=_pd_gold_prep(Xs); T=Xg.shape[1]; acc=np.full(T,np.nan); cls=np.unique(y); rng=np.random.default_rng(0)
        for t in range(T):
            sc,_=_pd_pca_feats(Xg[:,t,:]); feat=sc[:,:1]; idx=[]; ok=True
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
    _R1={w:{} for w in _OV_DESIGNS}; _R1RT={}
    for _e in _ev:
        for which in _OV_DESIGNS:
            des=_design(_ev[_e],which)
            if des is None: continue
            lbl,Xs,ys,nm,ch=des
            if nm<3: continue
            _R1[which][_e]=(_ev[_e]["ts"],_acc1(Xs,ys,nm)); _R1RT[_e]=_mrt(_e)

def _regs(secs,rt):
    return dict(Pre=(secs>=-4)&(secs<0),Post=(secs>=0)&(secs<=4),
               Baseline=(secs>=-4)&(secs<0),TOR=(secs>=0)&(secs<rt),Action=(secs>=rt)&(secs<=4))
def _p2(c,mA,mB,rng):
    idx=np.where(mA|mB)[0]; nB=int(mB.sum()); obs=np.nanmean(c[mB])-np.nanmean(c[mA])
    v=c[idx]; v=v[~np.isnan(v)]; a=np.arange(len(v)); null=np.empty(_OV_PERM)
    for p in range(_OV_PERM): rng.shuffle(a); null[p]=v[a[:nB]].mean()-v[a[nB:]].mean()
    return obs,float((np.sum(np.abs(null)>=abs(obs))+1)/(_OV_PERM+1))
def _st(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
def _cf(x,k=5):
    x=np.asarray(x,float); o=x.copy()
    for i in range(len(x)):
        seg=x[max(0,i-2):i+3]; seg=seg[~np.isnan(seg)]
        if len(seg): o[i]=seg.mean()
    return o

_PHCOL={"Baseline":"#4C6085","TOR":"#C89B3C","Action":"#A04A3C"}

def _overlay(ax, ts, acc, rt, ch, title):
    ax.plot(ts,_cf(acc),color="#333",lw=1.4,zorder=3)
    ax.axhline(ch,color="gray",lw=0.7,ls=":"); ax.axvline(0,color="k",lw=0.9,ls="--")
    if np.isfinite(rt): ax.axvline(rt,color="purple",lw=1.0,ls=":")
    R=_regs(ts,rt)
    # pre/post gray segments (thin, behind)
    for name,col,lw in [("Pre","#999",2.0),("Post","#999",2.0)]:
        m=R[name]
        if m.any(): ax.hlines(np.nanmean(acc[m]),ts[m].min(),ts[m].max(),color=col,lw=lw,ls="-",alpha=0.5,zorder=2)
    # phase coloured segments (thicker, on top)
    for name in ["Baseline","TOR","Action"]:
        m=R[name]
        if m.sum()>1:
            ax.hlines(np.nanmean(acc[m]),ts[m].min(),ts[m].max(),color=_PHCOL[name],lw=3,zorder=4)
    # stars: TOR vs Baseline, Action vs TOR (per-event)
    rng=np.random.default_rng(0)
    if R["TOR"].sum()>1 and R["Action"].sum()>1:
        _,pbt=_p2(acc,R["TOR"],R["Baseline"],rng); _,pta=_p2(acc,R["Action"],R["TOR"],rng)
        yb,yt,ya=[np.nanmean(acc[R[k]]) for k in ("Baseline","TOR","Action")]
        ax.text((ts[R["Baseline"]].max()+ts[R["TOR"]].min())/2,max(yb,yt)+0.02,_st(pbt),ha="center",fontsize=10,fontweight="bold")
        ax.text((ts[R["TOR"]].max()+ts[R["Action"]].min())/2,max(yt,ya)+0.02,_st(pta),ha="center",fontsize=10,fontweight="bold")
    ax.set_ylim(0,1); ax.set_ylabel("CV acc"); ax.set_title(title,fontsize=9); ax.grid(True,alpha=0.2)

for which in _OV_DESIGNS:
    ev_e=[e for e in _ev if e in _R1.get(which,{})]
    if not ev_e: continue
    n=len(ev_e)+1
    fig,axes=plt.subplots(n,1,figsize=(9,2.3*n),sharex=True); axes=np.atleast_1d(axes)
    fig.suptitle(f"{which} -- 1-dim PC1 LDA accuracy + region means (gray=pre/post, colour=phases)",fontsize=11,fontweight="bold")
    for k,_e in enumerate(ev_e):
        ts,acc=_R1[which][_e]; _overlay(axes[k],ts,acc,_R1RT[_e],0.5,_e)
    # average
    ref=_R1[which][ev_e[0]][0]
    def _al(t,c): return c if (len(t)==len(ref) and np.allclose(t,ref)) else np.interp(ref,t,c,left=np.nan,right=np.nan)
    avg=np.nanmean([_al(_R1[which][e][0],_R1[which][e][1]) for e in ev_e],0)
    _overlay(axes[-1],ref,avg,np.nanmean([_R1RT[e] for e in ev_e]),0.5,"AVERAGE")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0,0,1,0.97]); plt.show()
print("Overlay: horizontal region-mean segments on the 1-dim accuracy curve. Stars=per-event region-shuffle p (TOR vs Base, Action vs TOR).")
'''
ast.parse(code)
nb["cells"].insert(ins, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted 1-dim region-overlay plot cell at {ins} (before TFCE). Total: {len(nb['cells'])}.")
