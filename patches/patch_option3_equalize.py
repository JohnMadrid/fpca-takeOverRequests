import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

# Append a standalone Option-3 ED cell (equalized-channel ED), pre vs post,
# 3 events, reported next to plain ED. Self-contained-ish (uses _mf_arrs + _MF_* from A1).
cell = r'''# ============================================================
# OPTION 3: EQUALIZED-CHANNEL ED  (coordination-true dimensionality)
# Each channel's score block is normalised to unit total variance BEFORE
# the joint PCA, so no single channel dominates ED by amplitude. Then
# "low ED" = few SHARED dimensions across channels (the funnel claim),
# not single-channel dominance.
# Car-frame, per-window self-standardisation, lossless. Pre vs post, 3 events.
# Reports plain ED vs equalized ED + Opt1 cross-channel participation.
# Requires CELL A1 (_mf_arrs car-frame, _MF_*).
# ============================================================
import numpy as np

def _ed_from_jev(jev):
    jev=np.maximum(jev,0); s=jev.sum()
    return float((s**2)/np.sum(jev**2)) if s>0 else np.nan

def _participation(p):
    p=np.maximum(p,0); s=p.sum()
    if s<=0: return np.nan
    p=p/s; return 1.0/np.sum(p**2)

def _twostep_blocks(arr3d):
    """Per-channel univariate FPCA (lossless), per-window self unit-variance
    standardisation already assumed applied by caller. Returns list of score
    blocks (one per channel) + offsets."""
    M=arr3d.shape[2]; blocks=[]
    for ci in range(M):
        Xc=arr3d[:,:,ci]-arr3d[:,:,ci].mean(axis=0)
        cov=np.cov(Xc.T)
        ev,evec=np.linalg.eigh(cov)
        o=np.argsort(ev)[::-1]; ev=np.maximum(ev[o],0); evec=evec[:,o]
        nk=max(1,int(np.sum(ev>1e-12)))
        blocks.append(Xc@evec[:,:nk])
    return blocks

def _selfstd(arr3d):
    """Per-channel unit-variance over THIS window (drivers x time)."""
    M=arr3d.shape[2]; flat=arr3d.reshape(-1,M)
    mu=flat.mean(0); sd=flat.std(0,ddof=1); sd[sd<1e-12]=1.0
    return (arr3d-mu)/sd

def _ed_plain_and_equalized(arr3d):
    a=_selfstd(arr3d)
    blocks=_twostep_blocks(a)
    offs=np.cumsum([0]+[b.shape[1] for b in blocks])
    # plain: stack as-is
    Zp=np.hstack(blocks); Zp=Zp-Zp.mean(0)
    jev_p=np.maximum(np.linalg.eigh(np.cov(Zp.T))[0],0)
    ED_plain=_ed_from_jev(jev_p)
    # equalized: each block scaled to unit total variance before stacking
    eq=[]
    for b in blocks:
        tot=np.sum(np.var(b,axis=0,ddof=1))
        eq.append(b/np.sqrt(tot) if tot>1e-12 else b)
    Ze=np.hstack(eq); Ze=Ze-Ze.mean(0)
    jcov=np.cov(Ze.T)
    jev_e,jvec_e=np.linalg.eigh(jcov)
    o=np.argsort(jev_e)[::-1]; jev_e=np.maximum(jev_e[o],0); jvec_e=jvec_e[:,o]
    ED_eq=_ed_from_jev(jev_e)
    # Opt1 cross-channel participation on the EQUALIZED modes
    coord=[]
    for m in range(len(jev_e)):
        w=jvec_e[:,m]
        ce=np.array([np.sum(w[offs[c]:offs[c+1]]**2) for c in range(len(blocks))])
        coord.append(_participation(ce))
    wts=jev_e/jev_e.sum()
    xchan=float(np.nansum(wts*np.array(coord)))
    return ED_plain, ED_eq, xchan

print("OPTION 3: EQUALIZED-CHANNEL ED (car-frame, lossless)")
print("="*72)
print(f"{'event':<14}{'win':<5}{'ED plain':>10}{'ED equalized':>14}{'Opt1 xchan':>12}")
print("-"*55)
for _e in _MF_EVENTS:
    if _e not in _mf_arrs: continue
    for wname in ['pre','post']:
        arr=_mf_arrs[_e][wname]
        edp,ede,xc=_ed_plain_and_equalized(arr)
        print(f"{_MF_LABELS[_e]:<14}{wname:<5}{edp:>10.2f}{ede:>14.2f}{xc:>12.3f}")
print()
print("ED plain:     ED on self-standardised channels (current method).")
print("ED equalized: ED after forcing each channel equal total contribution")
print("              -> low ED now means few SHARED dims, not 1 channel dominating.")
print("Opt1 xchan:   cross-channel participation of the equalized modes (>1 = coordinated).")
'''
ast.parse(cell)
nb["cells"].append({"cell_type":"code","metadata":{},"execution_count":None,
                    "outputs":[],"source":cell})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Appended Option-3 equalized-ED cell. Total cells: {len(nb['cells'])}.")
