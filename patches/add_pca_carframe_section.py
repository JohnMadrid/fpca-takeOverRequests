import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

# avoid duplicate insert if rerun
already = any(("PCA -- CAR REFERENCE FRAME" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")))
              for c in nb["cells"])
assert not already, "PCA car-frame section already present; aborting to avoid duplicate."

md = r'''# PCA

Per-timepoint snapshot **PCA** effective dimensionality (the same method as the
"Expected ED Under Independence" cell above), re-run on the **car-reference-frame**
variables created later in this project. This is ordinary PCA at each timepoint,
not MFPCA.

Channels (car frame):
`NoseVectorCar.x`, `NoseVector.y`, `EyeDirCar.x`, `EyeDirWorldCombined.y`, `SteeringInput`.

The `.x` channels are de-rotated into the car frame; the `.y` (vertical) channels
are unaffected by car heading and are taken from the world columns, which inside
the car_reference files are byte-identical to their `*Car.y` counterparts. Input:
`data/cleaned_data/data_segment/car_reference/car_reference_{event}.csv`.
'''

code = r'''# ============================================================
# PCA -- CAR REFERENCE FRAME  (per-timepoint snapshot PCA ED)
# ------------------------------------------------------------
# Same pipeline as the "Expected ED Under Independence" cell, but on the
# car-frame variables and reading the car_reference CSVs. Original cell untouched.
#   ED(t) = (sum lambda)^2 / sum(lambda^2)  of the per-feature-standardised
#           (W drivers x M features) covariance at timepoint t.
# Crops to [-4, 4] s, shades post-onset region where ED falls below the
# pre-onset mean, plots per event + average, prints pre/post means + duration.
# ============================================================
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import os
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

# --- car-frame channels (.y taken from world cols == *Car.y inside these files) ---
modalities_car = [
    'NoseVectorCar.x',
    'NoseVector.y',
    'EyeDirCar.x',
    'EyeDirWorldCombined.y',
    'SteeringInput',
]
M_car = len(modalities_car)
_dir = os.getcwd()
_data_dir = _dir + '/data/'

_PCA_EVENTS = ['StagEventNew', 'FallingRocksEventNew', 'MotorcyclistEvent']
_EVT_LABELS = {'StagEventNew':'Stag crossing','MotorcyclistEvent':'Motorcyclist',
               'FallingRocksEventNew':'Falling rocks'}
_EVT_COLORS = {'StagEventNew':'steelblue','MotorcyclistEvent':'#d62728',
               'FallingRocksEventNew':'#2ca02c'}

def run_pca_ed_car(df_event):
    """Per-timepoint snapshot PCA ED curve on car-frame channels (all conditions
    pooled). Returns (ed_curve length T, W) or (None, 0)."""
    curves = []
    for _, sub in df_event.groupby('uid'):
        if len(sub) != points_per_window:
            continue
        data_scaled = StandardScaler().fit_transform(sub[modalities_car].values)
        curves.append(data_scaled.T)
    W = len(curves)
    if W == 0:
        return None, 0
    arr = np.stack(curves, axis=0)              # (W, M, T)
    arr = np.transpose(arr, (0, 2, 1))          # (W, T, M)
    secs = np.linspace(-5, 5, arr.shape[1])
    crop = (secs >= -4) & (secs <= 4)
    arr = arr[:, crop, :]
    _, T, _ = arr.shape
    ed_curve = np.zeros(T)
    for t in range(T):
        snap = arr[:, t, :].copy()
        snap -= snap.mean(axis=0)
        for m in range(M_car):
            sd = snap[:, m].std(ddof=1)
            if sd > 1e-12:
                snap[:, m] /= sd
        cov = snap.T @ snap / (W - 1)
        eig = eigh(cov, eigvals_only=True)
        p = eig / np.sum(eig)
        ed_curve[t] = 1.0 / np.sum(p**2)
    return ed_curve, W

# --- load each event from car_reference, run PCA ---
ed_curves_car_all  = []
ed_curves_car_dict = {}
events_loaded_car  = []
for evt in _PCA_EVENTS:
    fpath = _data_dir + f"cleaned_data/data_segment/car_reference/car_reference_{evt}.csv"
    if not os.path.exists(fpath):
        print(f"  WARNING: {evt} car_reference CSV not found, skipping"); continue
    df_evt = dd.read_csv(fpath, assume_missing=True, blocksize='100MB').compute()
    ed_curve, n_part = run_pca_ed_car(df_evt)
    if ed_curve is not None:
        ed_curves_car_all.append(ed_curve)
        ed_curves_car_dict[evt] = ed_curve
        events_loaded_car.append(evt)
        print(f"  Loaded {evt}: {n_part} participants")

n_events_car = len(events_loaded_car)
print(f"\nLoaded {n_events_car} events (car frame): {', '.join(events_loaded_car)}")

time_car  = np.linspace(-4, 4, len(ed_curves_car_all[0]))
pre_mask  = time_car < 0
post_mask = time_car >= 0

def _plot_ed_shaded_car(ax, t, ed, color, title):
    pre_mean = np.mean(ed[pre_mask])
    below = post_mask & (ed < pre_mean)
    ax.plot(t, ed, color=color, lw=1.5, label='Observed ED')
    ax.fill_between(t, ed, pre_mean, where=below, alpha=0.25, color='tomato',
                    label=f'Below pre-onset mean ({pre_mean:.2f})')
    ax.axhline(pre_mean, color='tomato', lw=1, ls='--', label=f'Pre-onset mean = {pre_mean:.2f}')
    ax.axvline(0, color='gray', ls='--', lw=0.9, alpha=0.7)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Effective Dimensionality')
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

def _print_stats_car(t, ed, label):
    dt = t[1]-t[0]
    pre_mean=np.mean(ed[pre_mask]); post_mean=np.mean(ed[post_mask])
    dur_below=(post_mask & (ed<pre_mean)).sum()*dt
    print(label)
    print(f"  Pre-onset mean ED       : {pre_mean:.4f}")
    print(f"  Post-onset mean ED      : {post_mean:.4f}")
    print(f"  Duration below pre-mean : {dur_below:.2f} s\n")

# --- per-event ---
for evt in events_loaded_car:
    fig, ax = plt.subplots(figsize=(8,5))
    _plot_ed_shaded_car(ax, time_car, ed_curves_car_dict[evt],
                        _EVT_COLORS.get(evt,'steelblue'),
                        _EVT_LABELS.get(evt,evt) + "  (car frame)")
    fig.tight_layout(); plt.show()
    _print_stats_car(time_car, ed_curves_car_dict[evt], _EVT_LABELS.get(evt,evt))

# --- average across events ---
ed_mean_car = np.mean(ed_curves_car_all, axis=0)
fig, ax = plt.subplots(figsize=(8,5))
_plot_ed_shaded_car(ax, time_car, ed_mean_car, 'steelblue',
                    f'Average across {n_events_car} events  (car frame)')
fig.tight_layout(); plt.show()
_print_stats_car(time_car, ed_mean_car, f'Average across {n_events_car} events (car frame)')
'''
ast.parse(code)

nb["cells"].append({"cell_type":"markdown","metadata":{},"source":md})
nb["cells"].append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Appended PCA car-frame section (markdown + code). Total cells: {len(nb['cells'])}.")
