import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already=any("LDA PRE/POST + S-V-F BASE/FULL STATS" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "stats cell present."

# insert after the LDA plot cell
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "LDA SHADED ACCURACY PLOTS" in "".join(c["source"]): ins=i
assert ins is not None

md = r'''## LDA statistics: pre vs post, and post S-vs-F in Base vs Full

Two label-shuffle permutation tests (recompute LDA inside the null):
1. **Pre vs post** per contrast: stat = mean(post acc) - mean(pre acc); null =
   per-driver pre/post time-reversal. Table of k/3 events with post>pre & p<.05,
   plus the average-curve result.
2. **Post S-vs-F: Base vs Full**: mean post accuracy of Success-vs-Failure decoding
   within Base-only vs Full-only drivers; null = shuffle Base/Full condition label.
Reuses cell-148 helpers (_ev, _build/_design/_acc_tc). Run cell 148 first.
'''

code = r'''# ============================================================
# LDA PRE/POST + S-V-F BASE/FULL STATS  (label-shuffle permutation)
# ------------------------------------------------------------
# Requires the TFCE-LDA car-frame cell helpers: _ev, _acc_tc, _design, _DESIGNS,
# _CF_VARS, _make_y (and _ev[event] = dict(ts,X,yw,yo)).
# ============================================================
import numpy as np
from tqdm.auto import tqdm
_ST_PERM=2000
_ST_RNG=np.random.default_rng(0)

def _prepost_stat(X,y,n_min,ts):
    acc,_=_acc_tc(X,y,n_min); pre=ts<0; post=ts>=0
    return np.nanmean(acc[post])-np.nanmean(acc[pre]), acc

def _perm_prepost(X,y,n_min,ts,P=_ST_PERM,rng=None):
    rng=rng or _ST_RNG
    obs,acc=_prepost_stat(X,y,n_min,ts); W=X.shape[0]; pre=ts<0; post=ts>=0
    null=np.empty(P)
    for p in range(P):
        flip=rng.random(W)<0.5
        Xp=X.copy(); Xp[flip,:,:]=Xp[flip][:,::-1,:]    # per-driver time-reversal
        accp,_=_acc_tc(Xp,y,n_min)
        null[p]=np.nanmean(accp[post])-np.nanmean(accp[pre])
    pv=(np.sum(null>=obs)+1)/(P+1)                       # one-tailed post>pre
    return obs,float(pv),acc

# ---------- (1) PRE vs POST per contrast (use the 4 balanced designs) ----------
_DES_USE=[d for d in _DESIGNS if d!="BvF_unbal" and d!="SvF_unbal"]
print("="*84); print("(1) PRE vs POST per contrast (post>pre, per-driver time-reversal null)"); print("="*84)
print(f"{'contrast':<26}{'event':<14}{'pre':>7}{'post':>7}{'post-pre':>9}{'p':>8}")
print("-"*84)
_st1={}; _acc_store={}
for which in _DES_USE:
    rows=[]
    for _e in tqdm(_ev,desc=which,leave=False):
        des=_design(_ev[_e],which)
        if des is None: continue
        lbl,Xs,ys,nm,ch=des; ts=_ev[_e]["ts"]
        if nm<3: continue
        obs,pv,acc=_perm_prepost(Xs,ys,nm,ts)
        pre=ts<0; post=ts>=0
        rows.append((_e,np.nanmean(acc[pre]),np.nanmean(acc[post]),obs,pv))
        _acc_store[(which,_e)]=(ts,acc)
        print(f"{lbl:<26}{_e:<14}{np.nanmean(acc[pre]):>7.3f}{np.nanmean(acc[post]):>7.3f}{obs:>+9.3f}{pv:>8.3f}{'*' if pv<.05 else ''}")
    _st1[which]=rows
    print("-"*84)

# k/3 agreement table + averaged-curve result
print("\nAGREEMENT (post>pre & p<.05) and AVERAGED-CURVE result:")
print(f"{'contrast':<26}{'k sig/total':>12}{'avg pre':>9}{'avg post':>10}{'avg d':>8}{'avg p':>8}")
print("-"*74)
for which in _DES_USE:
    rows=_st1[which]
    if not rows: print(f"{which:<26}{'no data':>12}"); continue
    k=sum(1 for _e,pr,po,d,p in rows if d>0 and p<.05); tot=len(rows)
    # averaged curve across events for this contrast, then perm on the avg
    accs=[_acc_store[(which,_e)][1] for _e,*_ in rows if (which,_e) in _acc_store]
    ts=_acc_store[(which,rows[0][0])][0]; pre=ts<0; post=ts>=0
    avg=np.nanmean(np.stack(accs),0)
    davg=np.nanmean(avg[post])-np.nanmean(avg[pre])
    # null for averaged curve: reuse per-event nulls is complex; quick proxy = shuffle pre/post of avg timepoints
    rng=np.random.default_rng(1); idx=np.where(pre|post)[0]; nA=int(pre.sum()); nullavg=np.empty(_ST_PERM)
    vals=avg[idx]
    for p in range(_ST_PERM):
        pm=rng.permutation(vals); nullavg[p]=pm[nA:].mean()-pm[:nA].mean()
    pavg=(np.sum(nullavg>=davg)+1)/(_ST_PERM+1)
    lbl=_design(_ev[rows[0][0]],which)[0]
    print(f"{lbl:<26}{f'{k}/{tot}':>12}{np.nanmean(avg[pre]):>9.3f}{np.nanmean(avg[post]):>10.3f}{davg:>+8.3f}{pavg:>8.3f}{'*' if pavg<.05 else ''}")

# ---------- (2) POST S-vs-F: Base only vs Full only ----------
print("\n"+"="*84)
print("(2) POST mean accuracy: S-vs-F decoding, Base only vs Full only (condition-label shuffle)")
print("="*84)
print(f"{'event':<14}{'post acc Base':>14}{'post acc Full':>14}{'Base-Full':>11}{'p':>8}")
print("-"*84)
for _e in tqdm(_ev,desc="SvF Base/Full",leave=False):
    d=_ev[_e]; ts=d["ts"]; post=ts>=0
    dbase=_design(d,"SvF_base"); dfull=_design(d,"SvF_full")
    if dbase is None or dfull is None: print(f"{_e:<14} missing"); continue
    _,Xb,yb,nmb,_=dbase; _,Xf,yf,nmf,_=dfull
    nm=min(nmb,nmf)
    if nm<3: print(f"{_e:<14} low n"); continue
    def _postacc(X,y):
        a,_=_acc_tc(X,y,nm); return np.nanmean(a[post])
    ab=_postacc(Xb,yb); af=_postacc(Xf,yf); obs=ab-af
    # null: pool Base+Full SvF drivers, reshuffle condition label, recompute both post accs
    # rebuild from raw via masks on the event
    yw=d["yw"]; yo=d["yo"]; succ=(yo==1); fail=(yo==0)
    base=(yw=="BaseCondition"); full=(yw=="FullLoopAR")
    poolB=np.where(base&(succ|fail))[0]; poolF=np.where(full&(succ|fail))[0]
    allidx=np.concatenate([poolB,poolF]); nB=len(poolB)
    rng=np.random.default_rng(7); null=np.empty(_ST_PERM)
    for p in range(_ST_PERM):
        perm=rng.permutation(allidx); gB=perm[:nB]; gF=perm[nB:]
        yB=yo[gB].astype(int); yF=yo[gF].astype(int)
        nm2=min(min((yB==0).sum(),(yB==1).sum()), min((yF==0).sum(),(yF==1).sum()))
        if nm2<3: null[p]=np.nan; continue
        aB,_=_acc_tc(d["X"][gB],yB,nm2); aF,_=_acc_tc(d["X"][gF],yF,nm2)
        null[p]=np.nanmean(aB[post])-np.nanmean(aF[post])
    null=null[~np.isnan(null)]
    pv=(np.sum(np.abs(null)>=abs(obs))+1)/(len(null)+1)
    print(f"{_e:<14}{ab:>14.3f}{af:>14.3f}{obs:>+11.3f}{pv:>8.3f}{'*' if pv<.05 else ''}")

print("\nNotes: (1) one-tailed post>pre, per-driver time-reversal null.")
print("       (2) two-sided, condition-label shuffle. %d perms each." % _ST_PERM)
'''
ast.parse(code)
nb["cells"].insert(ins+1,{"cell_type":"markdown","metadata":{},"source":md})
nb["cells"].insert(ins+2,{"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb,open(NB,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(f"Inserted LDA pre/post + SvF base/full stats cell after {ins}. Total: {len(nb['cells'])}.")
