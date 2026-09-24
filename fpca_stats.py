"""Iterable analysis code for Analysis_ED_PCA.ipynb.

Kept OUT of the notebook so it can be edited without forcing a "Revert File":
the notebook imports these functions with autoreload on, so editing this file
and re-running the wrapper cell is enough. No notebook-vs-disk conflict.

Every function takes the kernel objects it needs as arguments (arrays, the ED
curve fn, RT boundaries), so it has no hidden dependency on notebook globals.
"""
import numpy as np
from scipy import stats as _st

try:
    from tqdm.auto import tqdm as _tq
except Exception:  # tqdm optional
    def _tq(x, **k): return x


# default contrasts on (Baseline, TOR, Action); all sum to zero
CONTRASTS = {
    "Linear (decline)":   np.array([-1., 0., 1.]),   # net decline
    "Quadratic (hump)":   np.array([-1., 2., -1.]),  # non-monotonic peak (headline)
    "Rise (TOR-Base)":    np.array([-1., 1., 0.]),   # directional readout
    "Collapse (TOR-Act)": np.array([0., 1., -1.]),   # directional readout
}


def _star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def _phase_means(ed, secs, rt):
    mB = (secs >= -4) & (secs < 0)
    mT = (secs >= 0) & (secs < rt)
    mA = (secs >= rt) & (secs <= 4)
    return np.array([ed[mB].mean(), ed[mT].mean(), ed[mA].mean()])


def _jackknife_phase(arr, secs, rt, ed_curve, lab):
    """Tukey pseudo-values for the 3 phase means. Returns (plug-in, jackknife, NxM)."""
    N = arr.shape[0]
    full = _phase_means(ed_curve(arr), secs, rt)
    ps = np.empty((N, 3))
    for i in _tq(range(N), desc=f"jackknife {lab}"):
        thm = _phase_means(ed_curve(np.delete(arr, i, axis=0)), secs, rt)
        ps[i] = N * full - (N - 1) * thm
    return full, ps.mean(0), ps


def _rm_anova(ps):
    """One-way repeated-measures ANOVA on an (N x 3) pseudo-value matrix."""
    N = ps.shape[0]; g = ps.mean(); pm = ps.mean(0); sm = ps.mean(1)
    ssP = N * ((pm - g) ** 2).sum(); ssS = 3 * ((sm - g) ** 2).sum()
    ssT = ((ps - g) ** 2).sum(); ssE = ssT - ssP - ssS; dfE = 2 * (N - 1)
    F = (ssP / 2) / (ssE / dfE)
    return F, float(_st.f.sf(F, 2, dfE)), dfE


def jackknife_3phase(arrs_by_lab, secs, evk, rt_by_lab, ed_curve, contrasts=None):
    """Per-event jackknife 3-phase test (gold ED) + inverse-variance event-mean.

    arrs_by_lab : {label: (N,T,M)} raw per-driver curves per event
    secs        : shared time axis (cropped to [-4,4])
    evk         : ordered list of event labels
    rt_by_lab   : {label: RT boundary} (onset-specific, differs per event)
    ed_curve    : fn(arr)->ED curve (the gold recipe, e.g. _gp_ed_curve)
    Returns a results dict; also prints the tables.
    """
    C = contrasts or CONTRASTS
    est_by_c = {c: [] for c in C}; se_by_c = {c: [] for c in C}
    out = {"per_event": {}, "event_mean": {}}

    print("=" * 94)
    print("JACKKNIFE 3-PHASE  (gold ED, per event, onset-specific RT; pseudo-value one-sample t)")
    print("=" * 94)
    for lab in evk:
        arr = arrs_by_lab[lab]; rt = rt_by_lab[lab]; N = arr.shape[0]
        full, jk, ps = _jackknife_phase(arr, secs, rt, ed_curve, lab)
        F, pF, dfE = _rm_anova(ps)
        print(f"\n{lab}   (N={N}, RT={rt:.2f}s)   phase ED  "
              f"Base={full[0]:.3f}  TOR={full[1]:.3f}  Action={full[2]:.3f}")
        print(f"   RM-ANOVA phase effect: F(2,{dfE})={F:.2f}, p={pF:.4f} {_star(pF)}")
        print(f"   {'contrast':<20}{'est':>9}{'t':>9}{'p':>10}")
        ev = {"means": full, "F": F, "p_omni": pF, "contrasts": {}}
        for cname, w in C.items():
            s = ps @ w; est = float(jk @ w); t, p = _st.ttest_1samp(s, 0.0)
            se = s.std(ddof=1) / np.sqrt(N)
            est_by_c[cname].append(est); se_by_c[cname].append(se)
            ev["contrasts"][cname] = {"est": est, "t": float(t), "p": float(p), "se": se}
            print(f"   {cname:<20}{est:>+9.3f}{t:>9.2f}{p:>10.4f} {_star(p)}")
        out["per_event"][lab] = ev

    print("\n" + "=" * 94)
    print("EVENT-MEAN  (inverse-variance meta-analysis of the 3 per-event estimates)")
    print("  caveat: shared drivers across events -> mildly anticonservative; matched LMM is the rigorous upgrade")
    print("=" * 94)
    # pooled omnibus: Fisher's method on the per-event RM-ANOVA p-values
    _op = np.clip(np.array([out["per_event"][l]["p_omni"] for l in evk]), 1e-300, 1.0)
    _X2 = float(-2.0 * np.log(_op).sum()); _dff = 2 * len(_op)
    _pooled_omni = float(_st.chi2.sf(_X2, _dff))
    out["event_mean"]["_omnibus"] = {"X2": _X2, "df": _dff, "p": _pooled_omni}
    print(f"Pooled omnibus (Fisher, {len(_op)} events):  X2({_dff}) = {_X2:.1f},  p = {_pooled_omni:.2e} {_star(_pooled_omni)}")
    print("-" * 94)
    print(f"{'contrast':<20}{'pooled est':>12}{'z':>9}{'p':>11}")
    for cname in C:
        est = np.array(est_by_c[cname]); se = np.array(se_by_c[cname]); w = 1.0 / se ** 2
        e_bar = float((w * est).sum() / w.sum()); se_bar = float(np.sqrt(1.0 / w.sum()))
        z = e_bar / se_bar; p = float(2 * _st.norm.sf(abs(z)))
        out["event_mean"][cname] = {"est": e_bar, "z": z, "p": p}
        print(f"{cname:<20}{e_bar:>+12.3f}{z:>9.2f}{p:>11.4f} {_star(p)}")

    print("\nNotes:")
    print(" - Pseudo-value p_i = N*theta - (N-1)*theta_(-i); mean = bias-corrected (jackknife) estimate.")
    print(" - Quadratic [-1,2,-1] = non-monotonic hump (headline). Linear [-1,0,1] = net decline.")
    print(" - Per event -> no 3x pairwise family. Meta-analysis gives one event-mean p per contrast.")
    return out


# ============================================================
# Circular-shift permutation 3-phase test (the MAIN/group-level test).
# Per-event RT throughout (fixes the single mean-RT boundary the old cell used).
# Group-level statistic = variance of the 3 phase means (one-tailed lag null);
# contrasts get two-sided lag-null p's. AVERAGE = per-event phase means pooled
# across events (no shared mean-RT boundary anywhere).
# ============================================================
def _prep(arr):
    """Gold recipe scaling, roll-invariant: per-driver temporal demean + pooled per-channel SD."""
    A = arr.astype(float) - arr.astype(float).mean(axis=1, keepdims=True)
    M = A.shape[2]; psd = A.reshape(-1, M).std(0, ddof=1); psd[psd < 1e-12] = 1.0
    return A / psd


def _ed_from_X(X):
    """Participation-ratio ED(t) from prepped scores; vectorised over timepoints."""
    Sc = X - X.mean(0, keepdims=True); W = X.shape[0]
    cov = np.einsum('wtm,wtn->tmn', Sc, Sc) / (W - 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    return ev.sum(1) ** 2 / np.sum(ev ** 2, 1)


def _roll_pp(X, rng):
    """Independent random circular lag per driver (full wrap)."""
    W, T, _ = X.shape; lags = rng.integers(1, T, size=W)
    idx = (np.arange(T)[None, :] - lags[:, None]) % T
    return X[np.arange(W)[:, None], idx, :]


def _holm(praw):
    order = np.argsort(praw); m = len(praw); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (m - rank) * praw[idx])); adj[idx] = run
    return adj


def circshift_3phase(arrs_by_lab, secs, evk, rt_by_lab, ed_curve=None,
                     n_perm=5000, seed=0, contrasts=None):
    """Per-event circular-shift 3-phase permutation test (gold ED).

    Group-level p = variance of phase means, one-tailed lag null (the gate).
    Contrast p's = two-sided lag null. AVERAGE pools per-event (own-RT) phase means.
    ed_curve (optional) is used only to validate the fast ED against your pipeline.
    """
    C = contrasts or CONTRASTS
    secs = np.asarray(secs, float)
    _omni = lambda pm: float(np.var(pm))

    Xs = {}; masks = {}; obs_pm = {}
    for lab in evk:
        X = _prep(arrs_by_lab[lab]); Xs[lab] = X
        rt = rt_by_lab[lab]
        m = ((secs >= -4) & (secs < 0), (secs >= 0) & (secs < rt), (secs >= rt) & (secs <= 4))
        masks[lab] = m
        ed = _ed_from_X(X)
        if ed_curve is not None:
            ref = ed_curve(arrs_by_lab[lab])
            if not np.allclose(ed, ref, atol=1e-6, rtol=1e-4):
                print(f"  [warn] {lab}: fast ED vs ed_curve max diff {np.max(np.abs(ed-ref)):.2e}")
        obs_pm[lab] = np.array([ed[m[0]].mean(), ed[m[1]].mean(), ed[m[2]].mean()])

    obs_omni = {lab: _omni(obs_pm[lab]) for lab in evk}
    obs_con = {lab: {c: float(C[c] @ obs_pm[lab]) for c in C} for lab in evk}
    pm_avg = np.mean([obs_pm[lab] for lab in evk], 0)
    obs_omni_avg = _omni(pm_avg); obs_con_avg = {c: float(C[c] @ pm_avg) for c in C}

    rng = np.random.default_rng(seed)
    n_omni = {lab: np.empty(n_perm) for lab in evk}
    n_con = {lab: {c: np.empty(n_perm) for c in C} for lab in evk}
    n_omni_avg = np.empty(n_perm); n_con_avg = {c: np.empty(n_perm) for c in C}
    for p in _tq(range(n_perm), desc="circular-shift"):
        pms = []
        for lab in evk:
            ed = _ed_from_X(_roll_pp(Xs[lab], rng)); m = masks[lab]
            pm = np.array([ed[m[0]].mean(), ed[m[1]].mean(), ed[m[2]].mean()]); pms.append(pm)
            n_omni[lab][p] = _omni(pm)
            for c in C: n_con[lab][c][p] = C[c] @ pm
        pa = np.mean(pms, 0); n_omni_avg[p] = _omni(pa)
        for c in C: n_con_avg[c][p] = C[c] @ pa

    p_one = lambda o, nu: float((np.sum(nu >= o) + 1) / (n_perm + 1))
    p_two = lambda o, nu: float((np.sum(np.abs(nu) >= abs(o)) + 1) / (n_perm + 1))
    out = {"per_event": {}, "average": {}}

    print("=" * 94)
    print(f"CIRCULAR-SHIFT 3-PHASE  (gold ED, per-event RT; participant-level lag null; {n_perm} perms)")
    print("=" * 94)
    for lab in evk:
        pm = obs_pm[lab]; rt = rt_by_lab[lab]; N = arrs_by_lab[lab].shape[0]
        po = p_one(obs_omni[lab], n_omni[lab])
        print(f"\n{lab}   (N={N}, RT={rt:.2f}s)   phase ED  "
              f"Base={pm[0]:.3f}  TOR={pm[1]:.3f}  Action={pm[2]:.3f}")
        print(f"   OMNIBUS (variance of phase means, 1-tailed): stat={obs_omni[lab]:.4f}, p={po:.4f} {_star(po)}")
        print(f"   {'contrast':<20}{'est':>9}{'p':>10}")
        ev = {"means": pm, "rt": rt, "omni": obs_omni[lab], "p_omni": po, "contrasts": {}}
        for c in C:
            est = obs_con[lab][c]; pc = p_two(est, n_con[lab][c])
            ev["contrasts"][c] = {"est": est, "p": pc}
            print(f"   {c:<20}{est:>+9.3f}{pc:>10.4f} {_star(pc)}")
        out["per_event"][lab] = ev

    poA = p_one(obs_omni_avg, n_omni_avg)
    print(f"\nAVERAGE (per-event RT phase means, pooled across {len(evk)} events)   "
          f"Base={pm_avg[0]:.3f}  TOR={pm_avg[1]:.3f}  Action={pm_avg[2]:.3f}")
    print(f"   OMNIBUS (1-tailed): stat={obs_omni_avg:.4f}, p={poA:.4f} {_star(poA)}")
    print(f"   {'contrast':<20}{'est':>9}{'p':>10}")
    out["average"] = {"means": pm_avg, "omni": obs_omni_avg, "p_omni": poA, "contrasts": {}}
    for c in C:
        est = obs_con_avg[c]; pc = p_two(est, n_con_avg[c])
        out["average"]["contrasts"][c] = {"est": est, "p": pc}
        print(f"   {c:<20}{est:>+9.3f}{pc:>10.4f} {_star(pc)}")

    labs = list(evk); praw = np.array([out["per_event"][l]["p_omni"] for l in labs]); adj = _holm(praw)
    print("\nOmnibus gates, Holm-corrected across events:")
    for i, l in enumerate(labs):
        print(f"   {l:<16} p={praw[i]:.4f} -> Holm {adj[i]:.4f} {_star(adj[i])}")
    print("\nNotes:")
    print(" - Group-level gate = variance of the 3 phase means, one-tailed circular-shift (lag) null.")
    print(" - Contrasts two-sided; only 2 of 4 independent (Quad=Rise+Collapse, Lin=Rise-Collapse).")
    print(" - Per-event RT throughout; AVERAGE = per-event phase means pooled (no mean-RT boundary).")
    print(" - Gate on the omnibus (Holm x3); contrasts are the decomposition.")
    return out


def plot_phases(res, out=None, figsize=(11.0, 3.4)):
    """ED phase-mean plot (Baseline / Take-over request / Action) per event + average.
    Stars sit on the Rise (Base->TOR) and Collapse (TOR->Action) segments; omnibus p
    under each panel. Uses the corrected per-event-RT circular-shift results (res dict
    from circshift_3phase). No in-figure title (printed to stdout instead)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#E9EBE8", "axes.facecolor": "#E9EBE8",
                         "savefig.facecolor": "#E9EBE8"})
    items = list(res["per_event"].items()) + [("AVERAGE", res["average"])]
    phases = ["Baseline", "Take-over\nrequest", "Action"]; xp = [0, 1, 2]
    allm = np.concatenate([np.asarray(b["means"], float) for _, b in items])
    ylo, yhi = float(allm.min()) - 0.25, float(allm.max()) + 0.55
    fig, axs = plt.subplots(1, len(items), figsize=figsize, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (lab, b) in zip(axs, items):
        m = np.asarray(b["means"], float)
        ax.plot(xp, m, "-o", color="#37474F", lw=2, ms=8, mfc="#2055a8", mec="white", zorder=3)
        for xi, mi in zip(xp, m):
            ax.text(xi, mi + 0.07, f"{mi:.2f}", ha="center", fontsize=8, color="#333")
        rp = b["contrasts"]["Rise (TOR-Base)"]["p"]; cp = b["contrasts"]["Collapse (TOR-Act)"]["p"]
        ax.text(0.5, yhi - 0.13, _star(rp), ha="center", va="center", fontsize=11, color="#555")
        ax.text(1.5, yhi - 0.13, _star(cp), ha="center", va="center", fontsize=11, color="#555")
        ax.set_xticks(xp); ax.set_xticklabels(phases, fontsize=8)
        ax.set_ylim(ylo, yhi); ax.set_xlim(-0.4, 2.4); ax.grid(True, axis="y", alpha=0.3)
        rt = b.get("rt"); rtxt = f"RT={rt:.2f}s   " if rt is not None else ""
        po = b["p_omni"]; ptxt = "p<.001" if po < 0.001 else f"p={po:.3f}"
        ax.set_xlabel(f"{lab}\n{rtxt}omnibus {ptxt}", fontsize=9)
    axs[0].set_ylabel("Effective dimensionality")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    print("ED across phases (per-event-RT circular-shift; stars = Rise / Collapse segments, omnibus under each panel)")
    plt.show()
    return fig


def plot_phases_classic(res, out=None, figsize=(8.2, 5.4), stars=True, star_size=8.5, avg_stars=True):
    """Old-format single-panel phase-mean plot, fed the corrected per-event-RT
    results (res from circshift_3phase). Same layout as the original lag-null plot:
    real -4..4 time axis, phase-window centres, faint phase backgrounds, Okabe-Ito
    per-event lines, dashed dimmed average (no SEM ribbon), stars on the Base->TOR
    and TOR->Action segments, secondary top time axis. x-scaffold uses the mean RT
    for display; the plotted values/stars are per-event RT."""
    import matplotlib.pyplot as plt
    PCOL = {"Stag crossing": "#0072B2", "Falling rocks": "#009E73", "Motorcyclist": "#D55E00"}
    ev = res["per_event"]; avg = res["average"]
    rt_mean = float(np.nanmean([ev[l]["rt"] for l in ev]))
    xpos = [-2.0, rt_mean / 2.0, (rt_mean + 4.0) / 2.0]
    xlab = ["Baseline\n[-4, 0)", f"TOR\n[0, {rt_mean:.1f})", f"Action\n[{rt_mean:.1f}, 4]"]

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "figure.facecolor": "#E9EBE8", "axes.facecolor": "#E9EBE8",
                         "savefig.facecolor": "#E9EBE8"})
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvspan(-4, 0, color="#f2f2f2", zorder=0)
    ax.axvspan(0, rt_mean, color="#eaeaf2", zorder=0)
    ax.axvspan(rt_mean, 4, color="#f2eaea", zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.8, ls="--", zorder=1)
    ax.axvline(rt_mean, color="#bbbbbb", lw=0.8, ls=":", zorder=1)

    def seg_stars(ax, m, rp, cp, col, va, weight, size):
        for xa, xb, ya, yb, p in [(xpos[0], xpos[1], m[0], m[1], rp),
                                   (xpos[1], xpos[2], m[1], m[2], cp)]:
            ax.annotate(_star(p), ((xa + xb) / 2, (ya + yb) / 2), fontsize=size,
                        color=col, ha="center", va=va, fontweight=weight)

    for lab, b in ev.items():
        m = list(b["means"]); col = PCOL.get(lab, "#555")
        ax.plot(xpos, m, "o-", color=col, lw=1.8, ms=7, label=lab, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)
        if stars:
            seg_stars(ax, m, b["contrasts"]["Rise (TOR-Base)"]["p"],
                      b["contrasts"]["Collapse (TOR-Act)"]["p"], col, "bottom", "normal", star_size)

    m = list(avg["means"])
    ax.plot(xpos, m, ls="--", color="0.45", lw=1.8, alpha=0.5, zorder=6, marker="D", ms=7,
            markerfacecolor="0.45", markeredgecolor="white", markeredgewidth=1.0, label="Average")
    if stars and avg_stars:
        seg_stars(ax, m, avg["contrasts"]["Rise (TOR-Base)"]["p"],
                  avg["contrasts"]["Collapse (TOR-Act)"]["p"], "#444", "top", "bold", star_size)

    ax.set_xlim(-4, 4); ax.set_xticks(xpos); ax.set_xticklabels(xlab, fontsize=9.5)
    ax2 = ax.secondary_xaxis("top"); ax2.set_xticks([-4, -2, 0, 2, 4]); ax2.set_xlabel("Time from onset (s)", fontsize=9)
    ax.set_ylabel("Effective dimensionality (gold ED)")
    ax.set_title("ED across phases: Baseline / Take-over request / Action  (per-event RT)")
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    ax.legend(frameon=False, fontsize=9.5, loc="best")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Stars: *** p<.001  ** p<.01  * p<.05  ns.  Per-event-RT circular-shift; segments = Rise / Collapse.")
    plt.show()
    return fig


def phase_plots(res, out_dir=None):
    """STABLE entry point for the ED-across-phases figures. The notebook cell calls
    only this, so it never has to change (no more Revert File): edit the styling here
    and re-run the cell. Currently renders two versions:
      1) no significance stars
      2) larger stars, none on the average line
    """
    import os
    d = out_dir or (os.getcwd() + "/plots/")
    plot_phases_classic(res, stars=False, out=d + "ed_across_phases_nostars.png")
    plot_phases_classic(res, star_size=12, avg_stars=False, out=d + "ed_across_phases_stars.png")


def mfpca_routeA(arr, keep=None, prep=True):
    """Route-A MFPCA (Happ & Greven 2016, Sec 3.2), between-driver, one window.

    arr: (W drivers, T timepoints, M channels), raw. prep=True applies _prep
    (Operation C + pooled per-channel SD) -- the SAME standardisation as the
    snapshot ED, which serves as HG's weighted scalar product.

    Two-step: (1) univariate FPCA per channel (all components) -> scores;
    (2) stack scores, eigendecompose Z = Xi'Xi/(W-1) -> multivariate eigenvalues nu.
    functional ED = (sum nu)^2 / sum nu^2 (participation ratio, threshold-free).
    Also reconstructs the top-K space-time eigenfunctions and their energy profiles,
    and returns the Route-B (concatenate+PCA) eigenvalues for an equivalence receipt.
    """
    A = _prep(arr) if prep else np.asarray(arr, float)
    A = A - A.mean(0, keepdims=True)                 # cross-driver de-mean (E[X]=0)
    W, T, M = A.shape
    # step 1: univariate FPCA per channel (keep all components)
    phis, xis = [], []
    for m in range(M):
        U, S, Vt = np.linalg.svd(A[:, :, m], full_matrices=False)
        phis.append(Vt)                              # (r,T) orthonormal eigenfunctions
        xis.append(U * S)                            # (W,r) scores = Y @ V
    # step 2: stack scores -> multivariate covariance -> eigen
    Xi = np.concatenate(xis, axis=1)                 # (W, sum r)
    nu, C = np.linalg.eigh(Xi.T @ Xi / (W - 1))
    o = np.argsort(nu)[::-1]; nu = np.clip(nu[o], 0, None); C = C[:, o]
    ed = float(nu.sum() ** 2 / np.sum(nu ** 2))
    # top-K multivariate eigenfunctions psi_k^(m)(t) and energy e_k(t)=nu_k||psi_k(t)||^2
    K = keep if isinstance(keep, int) else int(np.ceil(ed)) + 1
    bnd = np.cumsum([0] + [p.shape[0] for p in phis])
    eigfun = np.zeros((K, M, T))
    for k in range(K):
        for m in range(M):
            eigfun[k, m] = C[bnd[m]:bnd[m + 1], k] @ phis[m]
    energy = nu[:K, None] * (eigfun ** 2).sum(1)     # (K,T), integrates to nu_k
    # Route-B receipt: concatenate standardised channels, one PCA
    sB = np.linalg.svd(A.reshape(W, T * M), compute_uv=False)
    nuB = np.sort(sB ** 2 / (W - 1))[::-1]
    r = min(len(nu), len(nuB))
    return dict(nu=nu, ed=ed, eigfun=eigfun, energy=energy, K=K, W=W, T=T, M=M,
                routeB_maxdiff=float(np.max(np.abs(nu[:r] - nuB[:r]))))
