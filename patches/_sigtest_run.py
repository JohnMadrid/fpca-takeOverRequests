import os
points_per_window=501
import matplotlib; matplotlib.use("Agg")
# ============================================================
# MFPCA  Q1  |  CELL A1: helpers + car-frame loader + raw-200 scree
# ------------------------------------------------------------
# Car-frame channels (NoseVectorCar.x/.y, EyeDirCar.x/.y, SteeringInput).
# Loader returns RAW per-driver arrays; standardisation is deferred to the
# decomposed sub-window (per-window self unit-variance) for pre/post comparability.
# Lossless by default (all non-zero components). Scree shows elbow + 95% markers.
# Requires cell 2 (outlier_features, points_per_window) + cell 4 (dtypes).
# ============================================================
import os, numpy as np, pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
from numpy.linalg import eigh as _eigh_mf

_MF_EVENTS = ['StagEventNew', 'FallingRocksEventNew', 'MotorcyclistEvent']
_MF_LABELS = {'StagEventNew':'Stag crossing','FallingRocksEventNew':'Falling rocks',
              'MotorcyclistEvent':'Motorcyclist'}
_MF_MODALITIES = ["NoseVectorCar.x","NoseVectorCar.y","EyeDirCar.x","EyeDirCar.y","SteeringInput"]
_MF_M = len(_MF_MODALITIES)
_MF_DATA_DIR = os.getcwd() + '/data/'

_MF_PRE_IDX  = np.arange(50, 250)     # t -4.00 .. -0.02
_MF_POST_IDX = np.arange(250, 450)    # t  0.00 .. +3.98
_MF_FULL_IDX = np.arange(50, 450)
_MF_NTP = 200
_MF_T_PRE  = np.linspace(-5,5,points_per_window)[_MF_PRE_IDX]
_MF_T_POST = np.linspace(-5,5,points_per_window)[_MF_POST_IDX]


def _mf_load_event(evt):
    """RAW per-driver (W,501,M) car-frame array. No standardisation here."""
    fpath = (_MF_DATA_DIR + "cleaned_data/data_segment/car_reference/"
             + f"car_reference_{evt}.csv")
    if not os.path.exists(fpath):
        print(f"  {evt}: not found"); return None
    df = dd.read_csv(fpath, assume_missing=True, blocksize='100MB').compute()
    curves = []
    for _, sub in df.groupby('uid'):
        if len(sub) != points_per_window:
            continue
        curves.append(sub.sort_values('time_from_event')[_MF_MODALITIES].values.astype(float))
    if not curves: return None
    return np.stack(curves, axis=0)


def _mf_ed(eigvals):
    ev = np.maximum(eigvals, 0); s = ev.sum()
    return float((s**2)/np.sum(ev**2)) if s > 0 else np.nan


def _mf_selfstd(arr3d):
    """Per-channel unit-variance over THIS sub-window (drivers x time)."""
    M = arr3d.shape[2]; flat = arr3d.reshape(-1, M)
    mu = flat.mean(0); sd = flat.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    return (arr3d - mu) / sd


def _mf_twostep_raw(arr3d, pve=None, fixed_k=None):
    """Self-standardise per window, per-channel PCA (lossless if pve>=1),
    stack -> joint PCA -> ED. Returns dict(ed, jev, jevec, per_ch, n_keep)."""
    arr3d = _mf_selfstd(arr3d.astype(float))
    W = arr3d.shape[0]
    per_ch, stacked = [], []
    for ci in range(_MF_M):
        Xc = arr3d[:, :, ci] - arr3d[:, :, ci].mean(axis=0)
        cov = np.cov(Xc.T)
        evals, evecs = _eigh_mf(cov)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0); evecs = evecs[:, order]
        if fixed_k is not None:
            nk = int(min(fixed_k, len(evals)))
        elif pve is None or pve >= 1.0:
            nk = int(np.sum(evals > 1e-12))
        else:
            pve_cum = np.cumsum(evals)/evals.sum()
            nk = int(np.searchsorted(pve_cum, pve) + 1)
        nk = max(1, min(nk, len(evals)))
        scores = Xc @ evecs[:, :nk]
        stacked.append(scores)
        per_ch.append(dict(evecs=evecs[:, :nk], n_keep=nk))
    Z = np.hstack(stacked); Zc = Z - Z.mean(0)
    jcov = np.cov(Zc.T)
    jev, jevec = _eigh_mf(jcov)
    order = np.argsort(jev)[::-1]
    jev = np.maximum(jev[order], 0); jevec = jevec[:, order]
    return dict(ed=_mf_ed(jev), jev=jev, jevec=jevec, per_ch=per_ch,
                n_keep=[p["n_keep"] for p in per_ch])


# ---- load events, build pre/post/full ----
print("Loading car-frame events (raw)...")
_mf_arrs = {}
for _e in _MF_EVENTS:
    arr = _mf_load_event(_e)
    if arr is None: continue
    _mf_arrs[_e] = dict(pre=arr[:, _MF_PRE_IDX, :], post=arr[:, _MF_POST_IDX, :],
                        full=arr[:, _MF_FULL_IDX, :])
    print(f"  {_MF_LABELS[_e]}: W={arr.shape[0]}")
assert _mf_arrs, "no events loaded"


# ---- scree (reference: first event, post window, self-standardised) ----
_scree_evt = _MF_EVENTS[0]
_scree_post = _mf_selfstd(_mf_arrs[_scree_evt]['post'])
_W_scree = _scree_post.shape[0]
print(f"\nScree: {_MF_LABELS[_scree_evt]} post (W={_W_scree})")
fig, axes = plt.subplots(1, _MF_M, figsize=(4*_MF_M, 3.6), squeeze=False)
fig.suptitle(f"Per-channel scree (car-frame) - {_MF_LABELS[_scree_evt]} post",
             fontsize=11, fontweight="bold")
print(f"{'channel':<20}{'#@90%':<8}{'#@95%':<8}{'#@99%':<8}{'elbow':<7}")
print("-"*51)
for ci in range(_MF_M):
    X = _scree_post[:, :, ci]; Xc = X - X.mean(0)
    ev = np.sort(np.maximum(_eigh_mf(np.cov(Xc.T))[0], 0))[::-1]
    pve = np.cumsum(ev)/ev.sum()
    k90=int(np.searchsorted(pve,.90)+1); k95=int(np.searchsorted(pve,.95)+1); k99=int(np.searchsorted(pve,.99)+1)
    xc=np.arange(1,len(ev)+1).astype(float)
    p1=np.array([xc[0],pve[0]]); p2=np.array([xc[-1],pve[-1]]); d=p2-p1; dn=d/(np.linalg.norm(d)+1e-12)
    vec=np.column_stack([xc,pve])-p1; proj=(vec@dn)[:,None]*dn[None,:]
    elbow=int(np.argmax(np.linalg.norm(vec-proj,axis=1)))
    print(f"{_MF_MODALITIES[ci]:<20}{k90:<8}{k95:<8}{k99:<8}{int(xc[elbow]):<7}")
    ax=axes[0,ci]; ax.plot(xc,pve,'o-',ms=2)
    ax.axhline(0.95,color='tomato',ls='--',lw=0.8)
    ax.plot(xc[elbow],pve[elbow],'o',color='#d62728',ms=6,label=f'elbow {int(xc[elbow])}')
    ax.plot(xc[k95-1],pve[k95-1],'s',color='#2ca02c',ms=6,label=f'95% {k95}')
    ax.set_title(_MF_MODALITIES[ci],fontsize=8); ax.set_ylim(0,1.02)
    ax.set_xlim(0,min(60,len(ev))); ax.grid(True,alpha=0.3); ax.legend(fontsize=6)
plt.tight_layout(rect=[0,0,1,0.92]); plt.close("all")
print("\nLossless by default. Scree shows elbow + 95% for reference only.")

# ============================================================
# COORDINATION TRIAL: Option 1 (cross-channel participation) and
# Option 2 (off-diagonal-only structure), vs ED. Car-frame data.
# Pre vs post, 3 events. Channel-level summary (one value per channel
# per timepoint via that channel's univariate score? No -- here we use a
# channel-summary covariance built from per-driver curves collapsed to
# a per-channel scalar per timepoint is wrong). Instead we operate on the
# JOINT stacked-score covariance (same as MFPCA two-step) and read
# coordination off the modes (Opt1) and off the cross-channel blocks (Opt2).
# Lossless (all components). No sig testing.
# Requires CELL A1 helpers (_mf_twostep_raw etc. + _mf_arrs car-frame).
# ============================================================
import numpy as np

_CO_EVENTS = _MF_EVENTS
_CO_LAB = _MF_LABELS
_CO_M = _MF_M                       # 5 channels
_CO_NK_CAP = None                  # lossless

def _co_selfstd(a):
    a=a.astype(float); M=a.shape[2]; f=a.reshape(-1,M)
    mu=f.mean(0); sd=f.std(0,ddof=1); sd[sd<1e-12]=1.0
    return (a-mu)/sd

def _co_blocks(arr3d):
    """Two-step to the stacked scores, KEEP per-channel block boundaries.
    Self-standardises per-window first (consistent with A1/group/Opt3)."""
    arr3d = _co_selfstd(arr3d)
    W = arr3d.shape[0]
    stacked = []; nks = []
    for ci in range(_CO_M):
        Xc = arr3d[:, :, ci] - arr3d[:, :, ci].mean(axis=0)
        cov = np.cov(Xc.T)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0); evecs = evecs[:, order]
        nk = int(np.sum(evals > 1e-12))          # lossless
        nk = max(1, nk)
        stacked.append(Xc @ evecs[:, :nk]); nks.append(nk)
    Z = np.hstack(stacked)
    offs = np.cumsum([0] + nks)
    return Z - Z.mean(axis=0), offs, nks

def _participation(p):
    p = np.maximum(p, 0); s = p.sum()
    if s <= 0: return np.nan
    p = p / s
    return 1.0 / np.sum(p**2)

def _ed(jev):
    jev = np.maximum(jev, 0); s = jev.sum()
    return float((s**2)/np.sum(jev**2)) if s > 0 else np.nan

def _opt1_crosschannel(arr3d):
    """For each joint mode, how much variance it puts on each CHANNEL
    (summed over that channel's score-axes). Score each mode by its
    cross-channel participation (1=single channel .. up to _CO_M). Weight
    by mode variance. Returns (coord_score, ED)."""
    Zc, offs, nks = _co_blocks(arr3d)
    jcov = np.cov(Zc.T)
    jev, jevec = np.linalg.eigh(jcov)
    order = np.argsort(jev)[::-1]
    jev = np.maximum(jev[order], 0); jevec = jevec[:, order]
    ED = _ed(jev)
    # per mode: channel-wise energy = sum of squared loadings within each channel block
    coord_per_mode = []
    for m in range(len(jev)):
        w = jevec[:, m]
        chan_energy = np.array([np.sum(w[offs[c]:offs[c+1]]**2) for c in range(_CO_M)])
        coord_per_mode.append(_participation(chan_energy))   # 1.._CO_M
    coord_per_mode = np.array(coord_per_mode)
    # variance-weighted mean cross-channel participation
    wts = jev / jev.sum()
    coord = float(np.nansum(wts * coord_per_mode))
    return coord, ED

def _opt2_offdiag(arr3d):
    """Coordination from the CHANNEL x CHANNEL correlation, self-variance
    removed. Build a per-driver per-channel summary = total score energy per
    channel? No -- we need cross-channel COVARIANCE over drivers. Use the
    leading univariate score of each channel as that channel's scalar summary
    per driver, then correlate channels across drivers. Coordination = mean
    absolute off-diagonal correlation (how much channels co-vary across
    drivers)."""
    arr3d = _co_selfstd(arr3d)
    # leading univariate score per channel (captures that channel's dominant
    # between-driver pattern) -> (W, M)
    feats = []
    for ci in range(_CO_M):
        Xc = arr3d[:, :, ci] - arr3d[:, :, ci].mean(axis=0)
        cov = np.cov(Xc.T)
        evals, evecs = np.linalg.eigh(cov)
        k = np.argmax(evals)
        feats.append(Xc @ evecs[:, k])
    F = np.column_stack(feats)              # (W, M)
    C = np.corrcoef(F.T)                    # M x M correlation across drivers
    iu = np.triu_indices(_CO_M, k=1)
    offdiag = np.abs(C[iu])
    return float(np.mean(offdiag)), C

# windows from A1
_CO_WIN = {'pre': _MF_PRE_IDX, 'post': _MF_POST_IDX}

print("COORDINATION TRIAL (car-frame, lossless)")
print("="*72)
print(f"{'event':<14}{'win':<5}{'ED':>8}{'Opt1 xchan':>12}{'Opt2 offdiag':>14}")
print("-"*54)
for _e in _CO_EVENTS:
    if _e not in _mf_arrs: continue
    for wname, widx in _CO_WIN.items():
        arr = _mf_arrs[_e]['pre'] if wname=='pre' else _mf_arrs[_e]['post']
        coord1, ED = _opt1_crosschannel(arr)
        coord2, _ = _opt2_offdiag(arr)
        print(f"{_CO_LAB[_e]:<14}{wname:<5}{ED:>8.2f}{coord1:>12.3f}{coord2:>14.3f}")
print()
print("Opt1 xchan: variance-weighted cross-channel participation (1=single-channel")
print("            modes .. higher=modes span more channels = more coordination).")
print("Opt2 offdiag: mean |between-channel correlation| of leading channel scores")
print("            (0=channels independent .. 1=fully coupled).")
print("Compare pre vs post: coordination should RISE post-onset if channels couple.")

# ============================================================
# COORDINATION SIG TESTING  (car-frame, lossless)
#  (1) pre vs post:  bootstrap 95% CI per metric + paired permutation p on Delta
#  (2) main effect in POST: Success vs Failure, observed diff + permutation p (equal-n)
#  (3) Delta(post-pre) by group: does the pre->post change differ S vs F?
# Metrics: ED, Opt1 (cross-channel participation), Opt2 (mean |off-diag corr|).
# All metrics are COHORT-level -> inference by resampling/permuting DRIVERS.
# Requires CELL A1 (_mf_arrs car-frame, _MF_*) + the COORDINATION TRIAL cell
# (_opt1_crosschannel, _opt2_offdiag, _co_selfstd).
# ============================================================
import numpy as np
import dask.dataframe as dd, os
from tqdm.auto import tqdm

_SIG_B    = 2000      # bootstrap reps
_SIG_PERM = 5000      # permutation reps
_SIG_SEED = 0
_rng_sig  = np.random.default_rng(_SIG_SEED)

# ---- the three cohort metrics, as functions of a (W, T, M) array ----
def _m_ED(a):   return _opt1_crosschannel(a)[1]
def _m_Opt1(a): return _opt1_crosschannel(a)[0]
def _m_Opt2(a): return _opt2_offdiag(a)[0]
_METRICS = [("ED", _m_ED), ("Opt1", _m_Opt1), ("Opt2", _m_Opt2)]

def _boot_ci(a, fn, B=_SIG_B, rng=None):
    rng = rng or _rng_sig; W = a.shape[0]
    vals = np.empty(B)
    for b in range(B):
        vals[b] = fn(a[rng.integers(0, W, W)])   # resample drivers w/ replacement
    return float(np.nanpercentile(vals,2.5)), float(np.nanpercentile(vals,97.5))

# ---------------- (1) PRE vs POST, paired permutation on the Delta ----------------
# Same drivers in pre and post. Build paired (W,T,M) pre & post (same driver order).
def _paired_pre_post(evt):
    pre = _mf_arrs[evt]['pre']; post = _mf_arrs[evt]['post']
    assert pre.shape[0]==post.shape[0]
    return pre, post

def _perm_pre_post(pre, post, fn, P=_SIG_PERM, rng=None):
    rng = rng or _rng_sig; W = pre.shape[0]
    obs = fn(post) - fn(pre)
    null = np.empty(P)
    for p in range(P):
        flip = rng.random(W) < 0.5           # per-driver swap of pre/post
        A = np.where(flip[:,None,None], post, pre)
        Bm = np.where(flip[:,None,None], pre, post)
        null[p] = fn(Bm) - fn(A)
    pval = (np.sum(np.abs(null) >= abs(obs)) + 1) / (P + 1)
    return obs, float(pval)

print("="*78)
print("(1) PRE vs POST  -  bootstrap 95% CI + paired permutation p on Delta(post-pre)")
print("="*78)
print(f"{'event':<14}{'metric':<7}{'pre':>7}{' [CI]':<16}{'post':>7}{' [CI]':<16}{'D':>7}{'p':>8}")
print("-"*78)
_sig1 = []
for _e in tqdm(_MF_EVENTS, desc="pre/post"):
    if _e not in _mf_arrs: continue
    pre, post = _paired_pre_post(_e)
    for mname, fn in _METRICS:
        vpre, vpost = fn(pre), fn(post)
        lpre, hpre = _boot_ci(pre, fn); lpost, hpost = _boot_ci(post, fn)
        d, pval = _perm_pre_post(pre, post, fn)
        star = "***" if pval<.001 else "**" if pval<.01 else "*" if pval<.05 else ""
        print(f"{_MF_LABELS[_e]:<14}{mname:<7}{vpre:>7.2f} [{lpre:5.2f},{hpre:5.2f}]"
              f"{vpost:>7.2f} [{lpost:5.2f},{hpost:5.2f}]{d:>+7.2f}{pval:>8.3f}{star}")
        _sig1.append(dict(event=_MF_LABELS[_e],metric=mname,pre=vpre,post=vpost,delta=d,p=pval))
    print("-"*78)

# ---------------- load group labels (Success/Failure) per event ----------------
_DATA_DIR = os.getcwd()+'/data/'
def _load_groups(evt):
    """Return success-mask & failure-mask aligned to _mf_arrs[evt] driver order.
    Re-load the csv in the SAME groupby order A1 used (groupby('uid'))."""
    fp=_DATA_DIR+f"cleaned_data/data_segment/car_reference/car_reference_{evt}.csv"
    df=dd.read_csv(fp,assume_missing=True,blocksize='100MB').compute()
    succ=[]
    for u,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        succ.append(sub['SuccessfulCompletionState'].iloc[0])
    succ=np.array(succ)
    return (succ==1.0), (succ==0.0)

# ---------------- (2) MAIN EFFECT IN POST: Success vs Failure (equal-n perm) ----------------
def _perm_groupdiff(arr, maskA, maskB, fn, P=_SIG_PERM, rng=None):
    """Equal-n: subsample larger group to n=min once for the OBSERVED stat is
    biased; instead use permutation of labels at full n with the metric computed
    on each label set (metric is n-sensitive, so we hold n equal by subsampling
    BOTH groups to common n in every draw, including observed)."""
    rng = rng or _rng_sig
    idxA = np.where(maskA)[0]; idxB = np.where(maskB)[0]
    n = min(len(idxA), len(idxB))
    if n < 5: return np.nan, np.nan, n
    # observed: average over R equal-n subsamples to stabilise
    R=200
    obsA=np.mean([fn(arr[rng.choice(idxA,n,replace=False)]) for _ in range(R)])
    obsB=np.mean([fn(arr[rng.choice(idxB,n,replace=False)]) for _ in range(R)])
    obs=obsA-obsB
    pool=np.concatenate([idxA,idxB])
    null=np.empty(P)
    for p in range(P):
        perm=rng.permutation(pool)
        gA=perm[:n]; gB=perm[len(idxA):len(idxA)+n]
        null[p]=fn(arr[gA])-fn(arr[gB])
    pval=(np.sum(np.abs(null)>=abs(obs))+1)/(P+1)
    return obs, float(pval), n, obsA, obsB

print()
print("="*78)
print("(2) MAIN EFFECT IN POST  -  Success vs Failure (equal-n, label permutation)")
print("="*78)
print(f"{'event':<14}{'metric':<7}{'n/grp':>6}{'Success':>9}{'Failure':>9}{'S-F':>8}{'p':>8}")
print("-"*78)
_sig2=[]
_grp_cache={}
for _e in tqdm(_MF_EVENTS, desc="post S/F"):
    if _e not in _mf_arrs: continue
    mS,mF=_load_groups(_e); _grp_cache[_e]=(mS,mF)
    post=_mf_arrs[_e]['post']
    for mname,fn in _METRICS:
        res=_perm_groupdiff(post,mS,mF,fn)
        if res[0] is np.nan or (isinstance(res[0],float) and np.isnan(res[0])):
            print(f"{_MF_LABELS[_e]:<14}{mname:<7}{'low n':>6}"); continue
        obs,pval,n,oS,oF=res
        star="***" if pval<.001 else "**" if pval<.01 else "*" if pval<.05 else ""
        print(f"{_MF_LABELS[_e]:<14}{mname:<7}{n:>6}{oS:>9.2f}{oF:>9.2f}{obs:>+8.2f}{pval:>8.3f}{star}")
        _sig2.append(dict(event=_MF_LABELS[_e],metric=mname,n=n,Success=oS,Failure=oF,diff=obs,p=pval))
    print("-"*78)

# ---------------- (3) Delta(post-pre) BY GROUP: does change differ S vs F? ----------------
def _group_delta(arr_pre, arr_post, mask, fn, R=200, rng=None):
    """Equal-n cohort delta for one group: mean over R subsamples of fn(post)-fn(pre)
    at common n. n is the group size (no cross-group balancing here, but report n)."""
    rng=rng or _rng_sig; idx=np.where(mask)[0]; n=len(idx)
    if n<5: return np.nan,n
    ds=[fn(arr_post[rng.choice(idx,n,replace=False)])-fn(arr_pre[rng.choice(idx,n,replace=False)])
        for _ in range(R)]
    return float(np.mean(ds)), n

def _perm_delta_groupdiff(arr_pre,arr_post,mA,mB,fn,P=_SIG_PERM,rng=None):
    """Test H0: Delta_A == Delta_B. Permute group labels; recompute (DeltaA-DeltaB)
    at equal n each draw."""
    rng=rng or _rng_sig
    iA=np.where(mA)[0]; iB=np.where(mB)[0]; n=min(len(iA),len(iB))
    if n<5: return np.nan,np.nan,n
    def _delta(idx):
        return fn(arr_post[rng.choice(idx,n,replace=False)])-fn(arr_pre[rng.choice(idx,n,replace=False)])
    R=200
    obs=np.mean([_delta(iA) for _ in range(R)])-np.mean([_delta(iB) for _ in range(R)])
    pool=np.concatenate([iA,iB]); null=np.empty(P)
    for p in range(P):
        perm=rng.permutation(pool); gA=perm[:n]; gB=perm[len(iA):len(iA)+n]
        null[p]=_delta(gA)-_delta(gB)
    pval=(np.sum(np.abs(null)>=abs(obs))+1)/(P+1)
    return float(obs),float(pval),n

print()
print("="*78)
print("(3) DELTA(post-pre) BY GROUP  -  Success vs Failure; p = does change differ?")
print("="*78)
print(f"{'event':<14}{'metric':<7}{'dS':>8}{'dF':>8}{'dS-dF':>8}{'p':>8}")
print("-"*78)
_sig3=[]
for _e in tqdm(_MF_EVENTS, desc="delta S/F"):
    if _e not in _mf_arrs: continue
    mS,mF=_grp_cache.get(_e, _load_groups(_e))
    pre,post=_mf_arrs[_e]['pre'],_mf_arrs[_e]['post']
    for mname,fn in _METRICS:
        dS,nS=_group_delta(pre,post,mS,fn)
        dF,nF=_group_delta(pre,post,mF,fn)
        obs,pval,n=_perm_delta_groupdiff(pre,post,mS,mF,fn)
        if np.isnan(obs):
            print(f"{_MF_LABELS[_e]:<14}{mname:<7}{'low n':>8}"); continue
        star="***" if pval<.001 else "**" if pval<.01 else "*" if pval<.05 else ""
        print(f"{_MF_LABELS[_e]:<14}{mname:<7}{dS:>+8.2f}{dF:>+8.2f}{obs:>+8.2f}{pval:>8.3f}{star}")
        _sig3.append(dict(event=_MF_LABELS[_e],metric=mname,dS=dS,dF=dF,diff=obs,p=pval))
    print("-"*78)

print("\\nNotes:")
print(" - CIs: 2000-rep driver bootstrap. p: permutation (5000 reps), two-sided.")
print(" - Group tests subsample BOTH groups to common n every draw (ED is n-sensitive).")
print(" - * p<.05  ** p<.01  *** p<.001. Results in _sig1/_sig2/_sig3.")
