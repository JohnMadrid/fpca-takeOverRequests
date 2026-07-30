import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

cell = r'''# ============================================================
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

def _co_blocks(arr3d):
    """Two-step to the stacked scores, but KEEP per-channel block boundaries.
    Returns Z (W, sum_nk), offsets (channel -> slice), and the per-channel
    n_keep list."""
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
'''
ast.parse(cell)
nb["cells"].append({"cell_type":"code","metadata":{},"execution_count":None,
                    "outputs":[],"source":cell})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Appended coordination trial cell. Total cells: {len(nb['cells'])}.")
