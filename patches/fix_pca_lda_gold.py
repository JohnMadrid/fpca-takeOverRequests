import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "LDA ON PCA COMPONENTS" in "".join(c["source"]): break
s="".join(nb["cells"][i]["source"])

# 1) replace per-timepoint feature fn with a GOLD-on-whole-curve preprocessing fn
old_fn='''def _pd_pca_feats(Xt):
    """Xt: (W, M) at one timepoint. Gold-standardise across drivers, PCA.
    Returns scores (W, M) in explained-variance order + PC1 cos^2 (M,)."""
    A=Xt.astype(float)
    A=A-A.mean(0)                                   # cross-driver center at t
    sd=A.std(0,ddof=1); sd[sd<1e-12]=1.0; A=A/sd    # pooled per-channel scale (per t)
    cov=np.cov(A.T)
    ev,evec=np.linalg.eigh(cov)
    o=np.argsort(ev)[::-1]; evec=evec[:,o]
    scores=A@evec                                   # (W, M) PC scores, PV order
    pc1cos2=evec[:,0]**2
    return scores, pc1cos2'''
new_fn='''def _pd_gold_prep(X):
    """GOLD preprocessing on the WHOLE curve, identical to the upstream gold ED
    cell: operation C (per-driver temporal demean) + pooled per-channel SD.
    X:(W,T,M) -> standardised (W,T,M)."""
    A=X.astype(float)
    A=A-A.mean(axis=1,keepdims=True)                # OPERATION C
    flat=A.reshape(-1,A.shape[2]); psd=flat.std(0,ddof=1); psd[psd<1e-12]=1.0
    return A/psd                                    # pooled per-channel scale

def _pd_pca_feats(At):
    """At: (W, M) at one timepoint, ALREADY gold-standardised. PCA across drivers.
    Returns scores (W,M) in PV order + PC1 cos^2 (M,)."""
    A=At-At.mean(0)                                 # cross-driver center at t
    cov=np.cov(A.T)
    ev,evec=np.linalg.eigh(cov)
    o=np.argsort(ev)[::-1]; evec=evec[:,o]
    return A@evec, evec[:,0]**2'''
assert old_fn in s, "feature fn not found"
s=s.replace(old_fn,new_fn,1)

# 2) in _pd_acc_curve, gold-prep the whole X once before the timepoint loop
old_acc='''def _pd_acc_curve(X, y, n_min, k_dims):
    """LDA CV accuracy over time using top-k_dims PCA scores. y has labels (0/1)."""
    T=X.shape[1]; acc=np.full(T,np.nan); pc1grid=np.full((_PD_M,T),np.nan)
    rng=np.random.default_rng(0); cls=np.unique(y)
    for t in range(T):
        sc,pc1=_pd_pca_feats(X[:,t,:]); pc1grid[:,t]=pc1'''
new_acc='''def _pd_acc_curve(X, y, n_min, k_dims):
    """LDA CV accuracy over time using top-k_dims PCA scores. y has labels (0/1).
    PCA is computed from the GOLD-standardised curve (matches upstream gold ED)."""
    Xg=_pd_gold_prep(X)                             # gold C + pooled scale, once
    T=Xg.shape[1]; acc=np.full(T,np.nan); pc1grid=np.full((_PD_M,T),np.nan)
    rng=np.random.default_rng(0); cls=np.unique(y)
    for t in range(T):
        sc,pc1=_pd_pca_feats(Xg[:,t,:]); pc1grid[:,t]=pc1'''
assert old_acc in s, "acc_curve head not found"
s=s.replace(old_acc,new_acc,1)

# 3) note: gold prep is per-subset? No -- PCA must be fit on the SAME cohort as
# upstream (all drivers in the contrast subset). _pd_acc_curve receives the
# contrast's X (Xs) so gold prep + PCA are on that subset, consistent.
ast.parse(s)
nb["cells"][i]["source"]=s
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {i}: PCA now uses gold pipeline (C + pooled), matching upstream ED. No per-timepoint scaling.")
