import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

# Find the markdown "Q1 pre vs post" anchor; insert rebuilt cells right after it,
# BEFORE the surviving trial/option3 cells.
anchor = None
for i, c in enumerate(nb["cells"]):
    s = "".join(c["source"]) if isinstance(c["source"], list) else c.get("source", "")
    if c["cell_type"] == "markdown" and "Q1 pre vs post" in s:
        anchor = i; break
if anchor is None:
    anchor = len(nb["cells"]) - 1

# ---------- A1: helpers + car-frame loader + scree ----------
A1 = r'''# ============================================================
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
plt.tight_layout(rect=[0,0,1,0.92]); plt.show()
print("\nLossless by default. Scree shows elbow + 95% for reference only.")
'''
ast.parse(A1)

# ---------- A2: sanity check (shuffle-ED vs n) ----------
A2 = r'''# ============================================================
# MFPCA  Q1  |  CELL A2: SANITY CHECK (shuffle-ED vs subgroup size n)
# Steep slope => ED scales with n => compare only at EQUAL n.
# Requires A1.  Lossless.
# ============================================================
_SANITY_NS = [10,15,20,40,80,160]
_SANITY_B  = 200
_rng_mf = np.random.default_rng(0)
_full_post = _mf_arrs[_scree_evt]['post']
_W_full = _full_post.shape[0]
_SANITY_NS = [n for n in _SANITY_NS if n <= _W_full]

def _mf_ed_subset(arr3d):
    return _mf_twostep_raw(arr3d)["ed"]

print(f"SANITY CHECK ({_MF_LABELS[_scree_evt]} post, lossless, B={_SANITY_B})")
_mean_ed,_lo_ed,_hi_ed=[],[],[]
for n in _SANITY_NS:
    eds=np.empty(_SANITY_B)
    for b in range(_SANITY_B):
        idx=_rng_mf.choice(_W_full,size=n,replace=False)
        eds[b]=_mf_ed_subset(_full_post[idx])
    _mean_ed.append(np.nanmean(eds)); _lo_ed.append(np.nanpercentile(eds,2.5)); _hi_ed.append(np.nanpercentile(eds,97.5))
    print(f"  n={n:4d}: ED={np.nanmean(eds):.2f} [{np.nanpercentile(eds,2.5):.2f},{np.nanpercentile(eds,97.5):.2f}]")
_obs_full=_mf_ed_subset(_full_post)
print(f"  full-sample ED (n={_W_full}): {_obs_full:.2f}")
fig,ax=plt.subplots(figsize=(7.5,4.5))
ax.plot(_SANITY_NS,_mean_ed,'o-',color="#333333",label="mean shuffle-ED")
ax.fill_between(_SANITY_NS,_lo_ed,_hi_ed,alpha=0.2,color="#333333",label="95% null band")
ax.axhline(_obs_full,color="#d62728",ls="--",lw=1.6,label=f"full-sample ED ({_obs_full:.2f})")
ax.set_xlabel("subgroup size n"); ax.set_ylabel("MFPCA ED")
ax.set_title("SANITY CHECK: ED vs n (lossless, car-frame)",fontsize=11,fontweight="bold")
ax.grid(True,alpha=0.3); ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print("\\nSteep => compare only at EQUAL n. Band width = resolution floor.")
'''
ast.parse(A2)

# ---------- Cell B: ED pre/post + spectra + eigenfunctions ----------
B = r'''# ============================================================
# MFPCA  Q1  |  CELL B: ED pre vs post + spectra + eigenfunctions
# Lossless, car-frame, per-window self-standardised. No sig testing.
# Requires A1.
# ============================================================
_PC_COLS=["#4C72B0","#DD8452","#55A868"]

def _mf_eigfuncs(res,n_modes=3):
    offs=np.cumsum([0]+res["n_keep"]); funcs=np.zeros((n_modes,_MF_M,_MF_NTP))
    for m in range(min(n_modes,res["jevec"].shape[1])):
        w=res["jevec"][:,m]
        for ci in range(_MF_M):
            funcs[m,ci]=res["per_ch"][ci]["evecs"]@w[offs[ci]:offs[ci+1]]
    return funcs

print("MFPCA (car-frame, lossless) - ED pre vs post")
_mf_results={}
for _e in _MF_EVENTS:
    if _e not in _mf_arrs: continue
    rp=_mf_twostep_raw(_mf_arrs[_e]['pre']); qp=_mf_twostep_raw(_mf_arrs[_e]['post'])
    _mf_results[_e]=dict(pre=rp,post=qp)
    for _tag,_r in [("pre",rp),("post",qp)]:
        _sh=_r['jev'][:5]/_r['jev'].sum()
        print(f"    [{_tag}] n_keep={_r['n_keep']} top5={np.round(_sh,3).tolist()} l1/tot={_sh[0]:.3f}")
    print(f"  {_MF_LABELS[_e]}: ED_pre={rp['ed']:.3f} ED_post={qp['ed']:.3f}")
    # spectrum
    fig,axes=plt.subplots(1,2,figsize=(13,4.2))
    fig.suptitle(f"{_MF_LABELS[_e]} - joint MFPCA spectrum (pre vs post)",fontsize=11,fontweight="bold")
    ns=min(15,len(rp['jev']),len(qp['jev'])); xr=np.arange(1,ns+1)
    axes[0].plot(xr,rp['jev'][:ns],'o-',color="#1f77b4",label="pre")
    axes[0].plot(xr,qp['jev'][:ns],'s-',color="#d62728",label="post")
    axes[0].set_title("raw eigenvalues",fontsize=10); axes[0].set_xlabel("component"); axes[0].set_ylabel("eigenvalue")
    axes[0].grid(True,alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].plot(xr,rp['jev'][:ns]/rp['jev'].sum(),'o-',color="#1f77b4",label="pre")
    axes[1].plot(xr,qp['jev'][:ns]/qp['jev'].sum(),'s-',color="#d62728",label="post")
    axes[1].set_title("normalised ratios",fontsize=10); axes[1].set_xlabel("component"); axes[1].set_ylabel("variance ratio")
    axes[1].grid(True,alpha=0.3); axes[1].legend(fontsize=8)
    fig.text(0.5,-0.02,f"ED_pre={rp['ed']:.3f}  ED_post={qp['ed']:.3f}",ha="center",fontsize=10)
    plt.tight_layout(rect=[0,0,1,0.93]); plt.show()
    # eigenfunctions
    ef_p,ef_q=_mf_eigfuncs(rp,3),_mf_eigfuncs(qp,3)
    fig,axes=plt.subplots(2,_MF_M,figsize=(3.2*_MF_M,6),squeeze=False)
    fig.suptitle(f"{_MF_LABELS[_e]} - top-3 joint eigenfunctions",fontsize=11,fontweight="bold")
    for row,(ef,ts,tag) in enumerate([(ef_p,_MF_T_PRE,"pre"),(ef_q,_MF_T_POST,"post")]):
        for ci in range(_MF_M):
            ax=axes[row,ci]
            for m in range(3): ax.plot(ts,ef[m,ci],color=_PC_COLS[m],lw=1.2,label=f"mode {m+1}" if ci==0 else None)
            ax.axhline(0,color="gray",lw=0.5)
            if row==0: ax.set_title(_MF_MODALITIES[ci],fontsize=8)
            if ci==0: ax.set_ylabel(tag,fontsize=9)
            ax.set_xlabel("t (s)"); ax.grid(True,alpha=0.3)
            if ci==0: ax.legend(fontsize=6,loc="upper right")
    plt.tight_layout(rect=[0,0,1,0.94]); plt.show()

print("\\nSummary (no sig testing):")
print(f"{'Event':<16}{'ED_pre':<9}{'ED_post':<9}{'drop':<8}")
print("-"*42)
for _e in _MF_EVENTS:
    if _e not in _mf_results: continue
    a=_mf_results[_e]['pre']['ed']; b=_mf_results[_e]['post']['ed']
    print(f"{_MF_LABELS[_e]:<16}{a:<9.3f}{b:<9.3f}{a-b:+.3f}")
'''
ast.parse(B)

# ---------- Group cell: S/F and Base/Full, pre/post/full, subsample ----------
GR = r'''# ============================================================
# MFPCA-ED by GROUP  (S vs F, Base vs Full) x pre/post/full
# Car-frame, per-window self-standardised, lossless.
# Subsample both groups to common n (R draws, no replacement) -> mean ED + 95% band.
# Requires A1.  No sig testing.
# ============================================================
_GR_R=200; _GR_SEED=0
_GR_DATA_DIR=os.getcwd()+'/data/'
_GR_MODALITIES=_MF_MODALITIES

def _gr_load(evt):
    fp=(_GR_DATA_DIR+"cleaned_data/data_segment/car_reference/"+f"car_reference_{evt}.csv")
    if not os.path.exists(fp): print(f"  {evt}: not found"); return None
    df=dd.read_csv(fp,assume_missing=True,blocksize='100MB').compute()
    suc=df.groupby('uid')['SuccessfulCompletionState'].first().to_dict()
    cond=df.groupby('uid')['ExperimentalCondition'].first().to_dict()
    arrs,outs,conds=[],[],[]
    for u,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        if suc.get(u) not in (0.0,1.0): continue
        arrs.append(sub.sort_values('time_from_event')[_GR_MODALITIES].values.astype(float))
        outs.append(int(suc[u])); conds.append(cond.get(u,'NA'))
    if not arrs: return None
    return dict(X=np.stack(arrs,0),out=np.array(outs),cond=np.array(conds))

def _gr_ed(Xfull,idx):
    return _mf_twostep_raw(Xfull[:,idx,:])["ed"]

def _gr_sub(Xfull,ma,mb,idx,R=_GR_R,seed=_GR_SEED):
    rng=np.random.default_rng(seed); n=min(len(ma),len(mb))
    if n<3: return (n,)+ (np.nan,)*6
    ea,eb=np.full(R,np.nan),np.full(R,np.nan)
    for r in range(R):
        ea[r]=_gr_ed(Xfull,idx) if False else _mf_twostep_raw(Xfull[rng.choice(ma,n,replace=False)][:,idx,:])["ed"]
        eb[r]=_mf_twostep_raw(Xfull[rng.choice(mb,n,replace=False)][:,idx,:])["ed"]
    return (n,np.nanmean(ea),np.nanpercentile(ea,2.5),np.nanpercentile(ea,97.5),
            np.nanmean(eb),np.nanpercentile(eb,2.5),np.nanpercentile(eb,97.5))

_GR_WIN={'pre':_MF_PRE_IDX,'post':_MF_POST_IDX,'full':_MF_FULL_IDX}
def _comparisons(d):
    out=d['out']; cond=d['cond']; base=(cond=='BaseCondition'); full=(cond=='FullLoopAR')
    return [("S vs F (all)","Success",np.where(out==1)[0],"Failure",np.where(out==0)[0]),
            ("Base vs Full","Base",np.where(base)[0],"Full",np.where(full)[0])]

print("MFPCA-ED by GROUP (car-frame, lossless, subsample R=%d)"%_GR_R)
print("="*70)
_gr_rows=[]
for _e in _MF_EVENTS:
    d=_gr_load(_e)
    if d is None: continue
    Xf=d['X']; print(f"\\n### {_MF_LABELS[_e]} (W={Xf.shape[0]})")
    for cname,la,ma,lb,mb in _comparisons(d):
        for wname,widx in _GR_WIN.items():
            n,mA,loA,hiA,mB,loB,hiB=_gr_sub(Xf,ma,mb,widx)
            flag="  <<low n" if n<12 else ""
            print(f"  {cname:<14}[{wname:<4}] n={n:<3} {la}={mA:5.2f}[{loA:4.1f},{hiA:4.1f}] {lb}={mB:5.2f}[{loB:4.1f},{hiB:4.1f}]{flag}")
            _gr_rows.append(dict(event=_MF_LABELS[_e],comp=cname,window=wname,n=n,
                                 grpA=la,edA=mA,loA=loA,hiA=hiA,grpB=lb,edB=mB,loB=loB,hiB=hiB))
_gr_df=pd.DataFrame(_gr_rows)
print("\\nDone. Results in _gr_df.")
'''
ast.parse(GR)

new_cells = []
for src in (A1, A2, B, GR):
    new_cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src})

for off, cell in enumerate(new_cells, start=1):
    nb["cells"].insert(anchor + off, cell)

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Rebuilt A1, A2, Cell B, Group cell -> inserted after cell {anchor}. "
      f"Total cells now: {len(nb['cells'])}.")
