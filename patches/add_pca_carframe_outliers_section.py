import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already = any("PCA CAR-FRAME (OUTLIERS REMOVED)" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source",""))
              for c in nb["cells"])
assert not already, "outliers-removed car-frame PCA section already present; aborting."

md = r'''## PCA (car frame, outliers removed)

Same per-timepoint snapshot **PCA** effective-dimensionality analysis and
permutation test as the PCA section above, but on the **DBSCAN
outliers-removed** inputs de-rotated into the car frame:
`data/cleaned_data/data_segment/car_reference/outliers_removed_5features/car_reference_{event}.csv`
(built by `make_car_reference_outliers_removed.py`).

Channels: `NoseVectorCar.x`, `NoseVector.y`, `EyeDirCar.x`, `EyeDirWorldCombined.y`, `SteeringInput`.
'''

# Single self-contained cell: ED curves + plots + permutation test, all on the
# outliers-removed car-frame files. Same method as the two PCA cells above.
code = r'''# ============================================================
# PCA CAR-FRAME (OUTLIERS REMOVED) -- snapshot PCA ED + permutation test
# ------------------------------------------------------------
# Same method as the PCA CAR REFERENCE FRAME cell and its permutation test,
# but reading the outliers-removed (DBSCAN nearest-neighbor) files de-rotated
# into the car frame. Self-contained.
#   ED(t) = (sum lambda)^2 / sum(lambda^2) of per-feature-standardised
#           (W x M) covariance at timepoint t; crop [-4,4]s.
#   Perm test: per-participant pre/post flip via time-axis reversal;
#   stat = mean(post ED) - mean(pre ED); one-tailed p.
# ============================================================
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import os
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

_N_PERM_OR = 5000
_RNG_OR    = np.random.default_rng(seed=0)
_data_dir_or = os.getcwd() + '/data/'
_SUBDIR_OR = "cleaned_data/data_segment/car_reference/outliers_removed_5features/"

_MODS_OR = ['NoseVectorCar.x','NoseVector.y','EyeDirCar.x','EyeDirWorldCombined.y','SteeringInput']
M_or = len(_MODS_OR)
_EVENTS_OR = ['StagEventNew','MotorcyclistEvent','FallingRocksEventNew']
_LAB_OR = {'StagEventNew':'Stag crossing','MotorcyclistEvent':'Motorcyclist',
           'FallingRocksEventNew':'Falling rocks'}
_COL_OR = {'StagEventNew':'steelblue','MotorcyclistEvent':'#d62728',
           'FallingRocksEventNew':'#2ca02c'}

def _ed_from_arr_or(arr):
    W, T, M = arr.shape
    ed = np.zeros(T)
    for t in range(T):
        snap = arr[:, t, :].copy(); snap -= snap.mean(axis=0)
        for m in range(M):
            s = snap[:, m].std(ddof=1)
            if s > 1e-12: snap[:, m] /= s
        cov = snap.T @ snap / (W - 1)
        eig = eigh(cov, eigvals_only=True)
        p = eig / eig.sum()
        ed[t] = 1.0 / np.sum(p ** 2)
    return ed

def _load_arr_or(evt):
    fpath = _data_dir_or + _SUBDIR_OR + f"car_reference_{evt}.csv"
    if not os.path.exists(fpath):
        return None, None
    df = dd.read_csv(fpath, assume_missing=True, blocksize='100MB').compute()
    curves = []
    for _, sub in df.groupby('uid'):
        if len(sub) != points_per_window: continue
        curves.append(StandardScaler().fit_transform(sub[_MODS_OR].values))
    if not curves: return None, None
    arr = np.stack(curves, axis=0)
    secs = np.linspace(-5, 5, arr.shape[1])
    crop = (secs >= -4) & (secs <= 4)
    return arr[:, crop, :], secs[crop]

def _perm_test_or(arr, secs, n_perm, rng, label):
    pre = secs < 0; post = secs >= 0; W, T, M = arr.shape
    obs_ed = _ed_from_arr_or(arr)
    obs_stat = np.mean(obs_ed[post]) - np.mean(obs_ed[pre])
    null_stats = np.zeros(n_perm)
    for p in range(n_perm):
        if p % 200 == 0: print(f"  {label}: perm {p}/{n_perm}")
        ap = arr.copy(); flip = rng.random(W) < 0.5
        ap[flip, :, :] = ap[flip][:, ::-1, :]
        ed_p = _ed_from_arr_or(ap)
        null_stats[p] = np.mean(ed_p[post]) - np.mean(ed_p[pre])
    return obs_ed, obs_stat, null_stats, np.mean(null_stats <= obs_stat)

def _plot_or(secs, obs_ed, obs_stat, null_stats, p_val, color, title):
    pre = secs < 0; post = secs >= 0; pre_mean = np.mean(obs_ed[pre])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(secs, obs_ed, color=color, lw=1.5, label='Observed ED')
    below = post & (obs_ed < pre_mean)
    ax.fill_between(secs, obs_ed, pre_mean, where=below, alpha=0.25, color='tomato',
                    label=f'Below pre mean ({pre_mean:.2f})')
    ax.axhline(pre_mean, color='tomato', lw=1, ls='--')
    ax.axvline(0, color='black', ls='--', lw=0.9, alpha=0.6)
    ax.set_xlabel('Time from event onset (s)'); ax.set_ylabel('Effective dimensionality')
    ax.set_title(title + '  (car frame, outliers removed)'); ax.legend(fontsize=8)
    ax.grid(True, ls=':', lw=0.6, color='#bbbbbb')
    ax2 = axes[1]
    ax2.hist(null_stats, bins=40, color='gray', alpha=0.7, edgecolor='none')
    ax2.axvline(obs_stat, color=color, lw=2, label=f'Observed = {obs_stat:.4f}\np = {p_val:.3f}')
    ax2.axvline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax2.set_xlabel('mean(post ED) - mean(pre ED)'); ax2.set_ylabel('Count')
    ax2.set_title(f'Null distribution  (n={len(null_stats)} perms)')
    ax2.legend(fontsize=9); ax2.grid(True, ls=':', lw=0.6, color='#bbbbbb')
    fig.tight_layout(); plt.show()
    post_mean = np.mean(obs_ed[post]); dur = below.sum() * (secs[1]-secs[0])
    print(f"{title}  (car frame, outliers removed)")
    print(f"  Pre-onset mean ED        : {pre_mean:.4f}")
    print(f"  Post-onset mean ED       : {post_mean:.4f}")
    print(f"  Duration below pre-mean  : {dur:.2f} s")
    print(f"  Test statistic (post-pre): {obs_stat:+.4f}")
    print(f"  p-value (one-tailed)     : {p_val:.3f}\n")

# ---- per-event ----
_all_or = []; _secs_or = None
for _evt in _EVENTS_OR:
    print(f"Loading {_evt} (car frame, outliers removed)...")
    _arr, _secs = _load_arr_or(_evt)
    if _arr is None: print("  WARNING: skipped"); continue
    print(f"  {_arr.shape[0]} participants")
    _all_or.append(_arr); _secs_or = _secs
    _oe,_ostat,_ns,_p = _perm_test_or(_arr,_secs,_N_PERM_OR,_RNG_OR,_LAB_OR[_evt])
    _plot_or(_secs,_oe,_ostat,_ns,_p,_COL_OR[_evt],_LAB_OR[_evt])

# ---- combined: pool participants ----
if len(_all_or) > 1:
    print("Combined permutation test (all events pooled)...")
    _ac = np.concatenate(_all_or, axis=0)
    _oe,_ostat,_ns,_p = _perm_test_or(_ac,_secs_or,_N_PERM_OR,_RNG_OR,'Combined')
    _plot_or(_secs_or,_oe,_ostat,_ns,_p,'#555555','Combined (all events pooled)')

# ---- combined: ED averaged across events ----
if len(_all_or) > 1:
    print("Combined permutation test (ED averaged across events)...")
    _oe_each=[]; _ns_each=[]
    for _ae in _all_or:
        _oe,_os,_ns,_ = _perm_test_or(_ae,_secs_or,_N_PERM_OR,_RNG_OR,'avg-loop')
        _oe_each.append(_oe); _ns_each.append(_ns)
    _oe_avg=np.mean(np.stack(_oe_each,0),0); _ns_avg=np.mean(np.stack(_ns_each,0),0)
    _pre=_secs_or<0; _post=_secs_or>=0
    _ostat_avg=np.mean(_oe_avg[_post])-np.mean(_oe_avg[_pre])
    _p_avg=np.mean(_ns_avg<=_ostat_avg)
    _plot_or(_secs_or,_oe_avg,_ostat_avg,_ns_avg,_p_avg,'#555555','Combined (ED averaged across events)')
'''
ast.parse(code)
nb["cells"].append({"cell_type":"markdown","metadata":{},"source":md})
nb["cells"].append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Appended PCA car-frame (outliers removed) section. Total cells: {len(nb['cells'])}.")
