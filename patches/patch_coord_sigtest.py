import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

sig_i = trial_i = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"]!="code": continue
    s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    if "COORDINATION SIG TESTING" in s: sig_i = i
    if "COORDINATION TRIAL" in s: trial_i = i
assert trial_i is not None, "trial cell not found"

cell = r'''# ============================================================
# COORDINATION SIG TESTING  (car-frame, lossless)
#   Each BETWEEN-GROUP comparison is run at EQUAL N (its two arms matched).
#   Different comparisons may use different N -- only the two arms WITHIN a
#   comparison must match, because ED/Opt1/Opt2 are sample-size sensitive.
#
#  (1) PRE vs POST: within-cohort paired test (full cohort, same drivers in
#      both windows); bootstrap 95% CI + paired permutation p on Delta.
#  (2a) MAIN EFFECT IN POST: Success vs Failure, balanced WITHIN condition
#       (outcome is confounded with condition), equal n per arm + perm p.
#  (2b) MAIN EFFECT IN POST: Base vs Full (two condition levels), equal n per
#       arm (= min(nBase,nFull)) + perm p.
#  (3a/3b) DELTA(post-pre) BY GROUP for each comparison: p = does the pre->post
#       change differ between the two arms (at that comparison's equal n).
#
# Metrics: ED, Opt1 (cross-channel participation), Opt2 (mean |off-diag corr|).
# Requires CELL A1 (_mf_arrs, _MF_*) + COORDINATION TRIAL cell
# (_opt1_crosschannel, _opt2_offdiag, _co_selfstd).
# ============================================================
import numpy as np
import dask.dataframe as dd, os
from tqdm.auto import tqdm

_SIG_B    = 2000      # bootstrap reps (pre/post CIs)
_SIG_PERM = 5000      # permutation reps
_SIG_R    = 200       # subsample reps to stabilise an equal-n observed/null stat
_SIG_SEED = 0
_rng_sig  = np.random.default_rng(_SIG_SEED)

def _m_ED(a):   return _opt1_crosschannel(a)[1]
def _m_Opt1(a): return _opt1_crosschannel(a)[0]
def _m_Opt2(a): return _opt2_offdiag(a)[0]
_METRICS = [("ED", _m_ED), ("Opt1", _m_Opt1), ("Opt2", _m_Opt2)]

# ================= labels (outcome + condition) per event =================
_DATA_DIR = os.getcwd()+'/data/'
def _load_labels(evt):
    """(succ_mask, cond_array) aligned to _mf_arrs driver order."""
    fp=_DATA_DIR+f"cleaned_data/data_segment/car_reference/car_reference_{evt}.csv"
    df=dd.read_csv(fp,assume_missing=True,blocksize='100MB').compute()
    succ=[]; cond=[]
    for u,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        succ.append(float(sub['SuccessfulCompletionState'].iloc[0]))
        cond.append(str(sub['ExperimentalCondition'].iloc[0]))
    return (np.array(succ)==1.0), np.array(cond)

_lab_cache={}
for _e in _MF_EVENTS:
    if _e in _mf_arrs: _lab_cache[_e]=_load_labels(_e)

# ================= (1) PRE vs POST  (within-cohort paired) =================
def _boot_ci(a, fn, B=_SIG_B, rng=None):
    rng=rng or _rng_sig; W=a.shape[0]; v=np.empty(B)
    for b in range(B): v[b]=fn(a[rng.integers(0,W,W)])
    return float(np.nanpercentile(v,2.5)), float(np.nanpercentile(v,97.5))

def _perm_pre_post(pre, post, fn, P=_SIG_PERM, rng=None):
    rng=rng or _rng_sig; W=pre.shape[0]; obs=fn(post)-fn(pre); null=np.empty(P)
    for p in range(P):
        flip=rng.random(W)<0.5
        A =np.where(flip[:,None,None], post, pre)
        Bm=np.where(flip[:,None,None], pre,  post)
        null[p]=fn(Bm)-fn(A)
    return obs, float((np.sum(np.abs(null)>=abs(obs))+1)/(P+1))

print("="*84)
print("(1) PRE vs POST  -  within-cohort paired; bootstrap 95% CI + permutation p")
print("="*84)
print(f"{'event':<14}{'metric':<7}{'N':>5}{'pre':>7}{' [CI]':<16}{'post':>7}{' [CI]':<16}{'D':>7}{'p':>8}")
print("-"*86)
_sig1=[]
for _e in tqdm(_MF_EVENTS, desc="pre/post"):
    if _e not in _mf_arrs: continue
    pre,post=_mf_arrs[_e]['pre'],_mf_arrs[_e]['post']; W=pre.shape[0]
    for mname,fn in _METRICS:
        vpre,vpost=fn(pre),fn(post)
        lpre,hpre=_boot_ci(pre,fn); lpost,hpost=_boot_ci(post,fn)
        d,pval=_perm_pre_post(pre,post,fn)
        star="***" if pval<.001 else "**" if pval<.01 else "*" if pval<.05 else ""
        print(f"{_MF_LABELS[_e]:<14}{mname:<7}{W:>5}{vpre:>7.2f} [{lpre:5.2f},{hpre:5.2f}]"
              f"{vpost:>7.2f} [{lpost:5.2f},{hpost:5.2f}]{d:>+7.2f}{pval:>8.3f}{star}")
        _sig1.append(dict(event=_MF_LABELS[_e],metric=mname,N=W,pre=vpre,post=vpost,delta=d,p=pval))
    print("-"*86)

# ================= equal-n machinery for between-group comparisons =================
# A "comparison" = (maskA, maskB, strata) where strata is an array used to
# balance composition (for S vs F we stratify by condition; for Base vs Full
# strata is constant -> simple equal-n).
def _equal_n(maskA, maskB, strata):
    idxA=np.where(maskA)[0]; idxB=np.where(maskB)[0]
    return sum(min(int(np.sum(strata[idxA]==s)), int(np.sum(strata[idxB]==s)))
               for s in np.unique(strata))

def _alloc(strata, idxA, idxB, target):
    levs=np.unique(strata)
    cap={s:min(int(np.sum(strata[idxA]==s)), int(np.sum(strata[idxB]==s))) for s in levs}
    tot=sum(cap.values())
    if tot<target: return None
    raw={s: target*cap[s]/tot for s in levs}
    base={s:int(np.floor(raw[s])) for s in levs}
    rem=target-sum(base.values())
    order=sorted(levs, key=lambda s: raw[s]-base[s], reverse=True); i=0
    while rem>0 and i<10*len(levs):
        s=order[i%len(order)]
        if base[s]<cap[s]: base[s]+=1; rem-=1
        i+=1
    return base

def _draw(maskA, maskB, strata, rng, target):
    idxA=np.where(maskA)[0]; idxB=np.where(maskB)[0]
    alloc=_alloc(strata, idxA, idxB, target)
    if alloc is None: return None,None
    selA=[]; selB=[]
    for s,k in alloc.items():
        if k==0: continue
        a=idxA[strata[idxA]==s]; b=idxB[strata[idxB]==s]
        selA.append(rng.choice(a,k,replace=False)); selB.append(rng.choice(b,k,replace=False))
    return np.concatenate(selA), np.concatenate(selB)

def _perm_labels(maskA, maskB, strata, rng):
    N=len(strata); pA=np.zeros(N,bool); pB=np.zeros(N,bool)
    idxA=np.where(maskA)[0]; idxB=np.where(maskB)[0]
    for s in np.unique(strata):
        pool=np.concatenate([idxA[strata[idxA]==s], idxB[strata[idxB]==s]])
        if len(pool)==0: continue
        nA=int(np.sum(strata[idxA]==s)); perm=rng.permutation(pool)
        pA[perm[:nA]]=True; pB[perm[nA:]]=True
    return pA,pB

def _grp_stat(arr, mA, mB, strata, fn, rng, N, R=_SIG_R):
    a=np.empty(R); b=np.empty(R)
    for r in range(R):
        sA,sB=_draw(mA,mB,strata,rng,N)
        if sA is None: return np.nan,np.nan,np.nan
        a[r]=fn(arr[sA]); b[r]=fn(arr[sB])
    return float(np.mean(a)-np.mean(b)), float(np.mean(a)), float(np.mean(b))

def _grp_test(arr, mA, mB, strata, fn, N, P=_SIG_PERM, rng=None):
    rng=rng or _rng_sig
    obs,gA,gB=_grp_stat(arr,mA,mB,strata,fn,rng,N)
    if np.isnan(obs): return None
    null=np.empty(P)
    for p in range(P):
        pA,pB=_perm_labels(mA,mB,strata,rng)
        sA,sB=_draw(pA,pB,strata,rng,N)
        null[p]=fn(arr[sA])-fn(arr[sB])
    return obs, float((np.sum(np.abs(null)>=abs(obs))+1)/(P+1)), gA, gB

def _grp_delta(pre, post, mA, mB, strata, fn, rng, N, which, R=_SIG_R):
    ds=np.empty(R)
    for r in range(R):
        sA,sB=_draw(mA,mB,strata,rng,N)
        if sA is None: return np.nan
        sel=sA if which=='A' else sB
        ds[r]=fn(post[sel])-fn(pre[sel])
    return float(np.mean(ds))

def _grp_delta_test(pre, post, mA, mB, strata, fn, N, P=_SIG_PERM, rng=None):
    rng=rng or _rng_sig
    dA=_grp_delta(pre,post,mA,mB,strata,fn,rng,N,'A')
    dB=_grp_delta(pre,post,mA,mB,strata,fn,rng,N,'B')
    if np.isnan(dA): return None
    obs=dA-dB; null=np.empty(P)
    for p in range(P):
        pA,pB=_perm_labels(mA,mB,strata,rng)
        sA,sB=_draw(pA,pB,strata,rng,N)
        null[p]=(fn(post[sA])-fn(pre[sA]))-(fn(post[sB])-fn(pre[sB]))
    return dA,dB,obs,float((np.sum(np.abs(null)>=abs(obs))+1)/(P+1))

# ---- build the two comparisons per event ----
def _comparisons(evt):
    succ,cond=_lab_cache[evt]
    out=[]
    # S vs F: stratify by condition
    out.append(("Success vs Failure","S","F", succ, ~succ, cond))
    # Base vs Full: condition levels, no sub-strata (constant strata)
    mBase=(cond=='BaseCondition'); mFull=(cond=='FullLoopAR')
    out.append(("Base vs Full","Base","Full", mBase, mFull, np.zeros(len(cond),int)))
    return out

# ================= (2) MAIN EFFECTS IN POST (each comparison at its own equal n) =================
print()
print("="*84)
print("(2) MAIN EFFECT IN POST  -  each comparison at EQUAL N per arm")
print("="*84)
print(f"{'event':<14}{'comparison':<20}{'metric':<7}{'N/arm':>6}{'A':>8}{'B':>8}{'A-B':>8}{'p':>8}")
print("-"*84)
_sig2=[]
for _e in tqdm(_MF_EVENTS, desc="post groups"):
    if _e not in _mf_arrs: continue
    post=_mf_arrs[_e]['post']
    for cname,la,lb,mA,mB,strata in _comparisons(_e):
        N=_equal_n(mA,mB,strata)
        for mname,fn in _METRICS:
            if N<5:
                print(f"{_MF_LABELS[_e]:<14}{cname:<20}{mname:<7}{'low n':>6}"); continue
            res=_grp_test(post,mA,mB,strata,fn,N)
            if res is None:
                print(f"{_MF_LABELS[_e]:<14}{cname:<20}{mname:<7}{'low n':>6}"); continue
            obs,pval,gA,gB=res
            star="***" if pval<.001 else "**" if pval<.01 else "*" if pval<.05 else ""
            print(f"{_MF_LABELS[_e]:<14}{cname:<20}{mname:<7}{N:>6}{gA:>8.2f}{gB:>8.2f}{obs:>+8.2f}{pval:>8.3f}{star}")
            _sig2.append(dict(event=_MF_LABELS[_e],comparison=cname,metric=mname,N=N,
                              A=gA,B=gB,Alab=la,Blab=lb,diff=obs,p=pval))
        print("-"*84)

# ================= (3) DELTA(post-pre) BY GROUP (each comparison at its own equal n) =================
print()
print("="*84)
print("(3) DELTA(post-pre) BY GROUP  -  p = does the pre->post change differ?")
print("="*84)
print(f"{'event':<14}{'comparison':<20}{'metric':<7}{'N/arm':>6}{'dA':>8}{'dB':>8}{'dA-dB':>8}{'p':>8}")
print("-"*84)
_sig3=[]
for _e in tqdm(_MF_EVENTS, desc="delta groups"):
    if _e not in _mf_arrs: continue
    pre,post=_mf_arrs[_e]['pre'],_mf_arrs[_e]['post']
    for cname,la,lb,mA,mB,strata in _comparisons(_e):
        N=_equal_n(mA,mB,strata)
        for mname,fn in _METRICS:
            if N<5:
                print(f"{_MF_LABELS[_e]:<14}{cname:<20}{mname:<7}{'low n':>6}"); continue
            res=_grp_delta_test(pre,post,mA,mB,strata,fn,N)
            if res is None:
                print(f"{_MF_LABELS[_e]:<14}{cname:<20}{mname:<7}{'low n':>6}"); continue
            dA,dB,obs,pval=res
            star="***" if pval<.001 else "**" if pval<.01 else "*" if pval<.05 else ""
            print(f"{_MF_LABELS[_e]:<14}{cname:<20}{mname:<7}{N:>6}{dA:>+8.2f}{dB:>+8.2f}{obs:>+8.2f}{pval:>8.3f}{star}")
            _sig3.append(dict(event=_MF_LABELS[_e],comparison=cname,metric=mname,N=N,
                              dA=dA,dB=dB,Alab=la,Blab=lb,diff=obs,p=pval))
        print("-"*84)

print("\\nNotes:")
print(" - Equal N WITHIN each comparison (two arms matched); comparisons may differ in N.")
print("   S vs F: stratified-balanced within ExperimentalCondition (outcome is")
print("   confounded with condition). Base vs Full: equal n = min(nBase,nFull).")
print(" - Permutation shuffles the group label (within strata for S vs F).")
print(f" - Observed group stats = mean over {_SIG_R} equal-n draws (ED is n-sensitive).")
print(f" - CIs: {_SIG_B}-rep bootstrap. p: {_SIG_PERM}-rep permutation, two-sided.")
print(" - * p<.05  ** p<.01  *** p<.001. Results in _sig1/_sig2/_sig3.")
'''
ast.parse(cell)
node={"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":cell}
if sig_i is not None:
    nb["cells"][sig_i]=node; where=f"replaced sig-test cell at {sig_i}"
else:
    nb["cells"].insert(trial_i+1, node); where=f"inserted after trial at {trial_i+1}"
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Per-comparison equal-N sig-test cell: {where}. Total cells: {len(nb['cells'])}.")
