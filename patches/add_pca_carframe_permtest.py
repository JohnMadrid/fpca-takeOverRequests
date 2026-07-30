import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

# insert AFTER the PCA car-frame code cell
pca_i = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"]=="code":
        s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if "PCA -- CAR REFERENCE FRAME" in s:
            pca_i = i; break
assert pca_i is not None, "PCA car-frame cell not found"

already = any("PCA CAR-FRAME -- PERMUTATION TEST" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source",""))
              for c in nb["cells"])
assert not already, "perm-test cell already present; aborting."

# Same permutation test as the original cell 35
# ("Permutation test: is post-onset ED significantly below pre-onset mean?"),
# applied to the car-frame channels + car_reference CSVs (the PCA cell above).
code = r'''# ============================================================
# PCA CAR-FRAME -- PERMUTATION TEST
#   Is post-onset ED significantly below the pre-onset mean?
# ------------------------------------------------------------
# Identical method to the original cell
#   "Permutation test: is post-onset ED significantly below pre-onset mean?"
# but on the car-frame channels (modalities_car) and the car_reference CSVs.
#
# Null: for each participant independently, flip pre/post with p=0.5 by reversing
# the time axis -- destroys time-locked structure, preserves within-participant
# covariance; under H0 pre and post are exchangeable.
# Statistic: mean(ED[post]) - mean(ED[pre])  (negative = post dips below pre).
# One-tailed p = fraction of null stats <= observed. Per event + two combined
# variants (participants pooled; ED averaged across events).
# Requires the PCA CAR REFERENCE FRAME cell above (modalities_car, M_car,
# run_pca_ed_car helpers are NOT reused -- this cell is self-contained).
# ============================================================
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
import os
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

_N_PERM_C  = 5000
_RNG_C     = np.random.default_rng(seed=0)
_data_dir_c = os.getcwd() + '/data/'

# car-frame channels (fall back to a local def if the PCA cell wasn't run)
try:
    _MODS_C = list(modalities_car)
except NameError:
    _MODS_C = ['NoseVectorCar.x','NoseVector.y','EyeDirCar.x','EyeDirWorldCombined.y','SteeringInput']

_EVENTS_C = ['StagEventNew', 'MotorcyclistEvent', 'FallingRocksEventNew']
_EVT_LAB_C = {'StagEventNew':'Stag crossing','MotorcyclistEvent':'Motorcyclist',
              'FallingRocksEventNew':'Falling rocks'}
_EVT_COL_C = {'StagEventNew':'steelblue','MotorcyclistEvent':'#d62728',
              'FallingRocksEventNew':'#2ca02c'}

def _ed_from_arr_c(arr):
    """arr: (W, T, M). Returns ED curve (T,)."""
    W, T, M = arr.shape
    ed = np.zeros(T)
    for t in range(T):
        snap = arr[:, t, :].copy()
        snap -= snap.mean(axis=0)
        for m in range(M):
            s = snap[:, m].std(ddof=1)
            if s > 1e-12:
                snap[:, m] /= s
        cov = snap.T @ snap / (W - 1)
        eig = eigh(cov, eigvals_only=True)
        p   = eig / eig.sum()
        ed[t] = 1.0 / np.sum(p ** 2)
    return ed

def _load_arr_c(evt):
    """car_reference CSV, StandardScaler per participant. Returns (W,T_crop,M), secs_crop."""
    fpath = _data_dir_c + f"cleaned_data/data_segment/car_reference/car_reference_{evt}.csv"
    if not os.path.exists(fpath):
        return None, None
    df = dd.read_csv(fpath, assume_missing=True, blocksize='100MB').compute()
    curves = []
    for _, sub in df.groupby('uid'):
        if len(sub) != points_per_window:
            continue
        curves.append(StandardScaler().fit_transform(sub[_MODS_C].values))   # (T, M)
    if not curves:
        return None, None
    arr  = np.stack(curves, axis=0)
    secs = np.linspace(-5, 5, arr.shape[1])
    crop = (secs >= -4) & (secs <= 4)
    return arr[:, crop, :], secs[crop]

def _perm_test_c(arr, secs, n_perm, rng, label):
    pre_mask  = secs < 0
    post_mask = secs >= 0
    W, T, M = arr.shape
    obs_ed   = _ed_from_arr_c(arr)
    obs_stat = np.mean(obs_ed[post_mask]) - np.mean(obs_ed[pre_mask])
    null_stats  = np.zeros(n_perm)
    null_curves = np.zeros((n_perm, T))
    for p in range(n_perm):
        if p % 200 == 0:
            print(f"  {label}: perm {p}/{n_perm}")
        arr_perm = arr.copy()
        flip = rng.random(W) < 0.5
        arr_perm[flip, :, :] = arr_perm[flip][:, ::-1, :]   # reverse time axis
        ed_perm = _ed_from_arr_c(arr_perm)
        null_curves[p] = ed_perm
        null_stats[p]  = np.mean(ed_perm[post_mask]) - np.mean(ed_perm[pre_mask])
    p_val = np.mean(null_stats <= obs_stat)   # one-tailed (post below pre)
    return obs_ed, obs_stat, null_stats, null_curves, p_val

def _plot_result_c(secs, obs_ed, obs_stat, null_stats, p_val, color, title):
    pre_mask  = secs < 0
    post_mask = secs >= 0
    pre_mean  = np.mean(obs_ed[pre_mask])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(secs, obs_ed, color=color, linewidth=1.5, label='Observed ED')
    below = post_mask & (obs_ed < pre_mean)
    ax.fill_between(secs, obs_ed, pre_mean, where=below, alpha=0.25, color='tomato',
                    label=f'Below pre mean ({pre_mean:.2f})')
    ax.axhline(pre_mean, color='tomato', linewidth=1, linestyle='--')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.9, alpha=0.6)
    ax.set_xlabel('Time from event onset (s)'); ax.set_ylabel('Effective dimensionality')
    ax.set_title(title + '  (car frame)'); ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', linewidth=0.6, color='#bbbbbb')
    ax2 = axes[1]
    ax2.hist(null_stats, bins=40, color='gray', alpha=0.7, edgecolor='none')
    ax2.axvline(obs_stat, color=color, linewidth=2,
                label=f'Observed = {obs_stat:.4f}\np = {p_val:.3f}')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.set_xlabel('mean(post ED) - mean(pre ED)'); ax2.set_ylabel('Count')
    ax2.set_title(f'Null distribution  (n={len(null_stats)} perms)')
    ax2.legend(fontsize=9); ax2.grid(True, linestyle=':', linewidth=0.6, color='#bbbbbb')
    fig.tight_layout(); plt.show()
    post_mean = np.mean(obs_ed[post_mask])
    below_dur = below.sum() * (secs[1] - secs[0])
    print(f"{title}  (car frame)")
    print(f"  Pre-onset mean ED        : {pre_mean:.4f}")
    print(f"  Post-onset mean ED       : {post_mean:.4f}")
    print(f"  Duration below pre-mean  : {below_dur:.2f} s")
    print(f"  Test statistic (post-pre): {obs_stat:+.4f}")
    print(f"  p-value (one-tailed)     : {p_val:.3f}\n")

# ---- per-event ----
_all_arrs_c = []
_secs_c = None
for _evt in _EVENTS_C:
    print(f"Loading {_evt} (car frame)...")
    _arr, _secs = _load_arr_c(_evt)
    if _arr is None:
        print("  WARNING: skipped"); continue
    print(f"  {_arr.shape[0]} participants")
    _all_arrs_c.append(_arr); _secs_c = _secs
    _oe,_ostat,_ns,_nc,_p = _perm_test_c(_arr,_secs,_N_PERM_C,_RNG_C,_EVT_LAB_C[_evt])
    _plot_result_c(_secs,_oe,_ostat,_ns,_p,_EVT_COL_C[_evt],_EVT_LAB_C[_evt])

# ---- combined: pool all participants ----
if len(_all_arrs_c) > 1:
    print("Combined permutation test (all events pooled)...")
    _arr_comb = np.concatenate(_all_arrs_c, axis=0)
    _oe,_ostat,_ns,_nc,_p = _perm_test_c(_arr_comb,_secs_c,_N_PERM_C,_RNG_C,'Combined')
    _plot_result_c(_secs_c,_oe,_ostat,_ns,_p,'#555555','Combined (all events pooled)')

# ---- combined: ED averaged across events ----
if len(_all_arrs_c) > 1:
    print("Combined permutation test (ED averaged across events)...")
    _oe_each=[]; _ns_each=[]
    for _arr_e in _all_arrs_c:
        _oe,_os,_ns,_nc,_ = _perm_test_c(_arr_e,_secs_c,_N_PERM_C,_RNG_C,'avg-loop')
        _oe_each.append(_oe); _ns_each.append(_ns)
    _oe_avg = np.mean(np.stack(_oe_each,0),0)
    _ns_avg = np.mean(np.stack(_ns_each,0),0)
    _pre_a=_secs_c<0; _post_a=_secs_c>=0
    _ostat_avg = np.mean(_oe_avg[_post_a]) - np.mean(_oe_avg[_pre_a])
    _p_avg = np.mean(_ns_avg <= _ostat_avg)
    _plot_result_c(_secs_c,_oe_avg,_ostat_avg,_ns_avg,_p_avg,'#555555',
                   'Combined (ED averaged across events)')
'''
ast.parse(code)
nb["cells"].insert(pca_i+1, {"cell_type":"code","metadata":{},"execution_count":None,
                             "outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted PCA car-frame permutation-test cell after PCA cell {pca_i}. Total: {len(nb['cells'])}.")
