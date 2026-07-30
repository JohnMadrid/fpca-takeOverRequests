import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

already = any("REGION-LABEL PERMUTATION TEST" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source",""))
              for c in nb["cells"])
assert not already, "region perm cell already present."

ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "REGION MEANS -- two contrasts" in "".join(c["source"]): ins=i
assert ins is not None, "gold ED region-means cell not found"

md = r'''### Region-label permutation test (two contrasts)

Both contrasts tested by shuffling region labels over the ED(t) timepoints
(region sizes preserved), recomputing the contrast statistic, two-sided p.

- **Contrast 1 (pre vs post):** stat = mean(Post) - mean(Pre).
- **Contrast 2 (3 phases: Baseline / TOR / Action):** three pairwise mean
  differences each with its own p, plus an omnibus (variance across the three
  phase means = "any phase differs").

TOR/Action boundary = mean RT_Steering per event (option A, fixed across drivers).
Reuses _gold_ED_store from the gold ED cell above.
'''

code = r'''# ============================================================
# REGION-LABEL PERMUTATION TEST  (gold ED, two contrasts)
# ------------------------------------------------------------
# Null: pool the timepoints of the relevant window, randomly reassign region
# labels keeping each region's timepoint count fixed, recompute the statistic.
# Two-sided p. Reuses _gold_ED_store (secs, ed, rt) from the gold ED cell.
# ============================================================
import numpy as np

_RP_PERM = 5000
_RP_RNG  = np.random.default_rng(0)

def _regions_rp(secs, rt):
    return dict(
        Pre =(secs>=-4)&(secs<0),
        Post=(secs>=0)&(secs<=4),
        Baseline=(secs>=-4)&(secs<0),
        TOR =(secs>=0)&(secs<rt),
        Action=(secs>=rt)&(secs<=4),
    )

def _perm_two_region(ed, maskA, maskB, P=_RP_PERM, rng=None):
    """Stat = mean(B)-mean(A). Shuffle labels over the union of A,B timepoints."""
    rng=rng or _RP_RNG
    idx=np.where(maskA|maskB)[0]; nA=int(maskA.sum())
    obs=ed[maskB].mean()-ed[maskA].mean()
    vals=ed[idx]; null=np.empty(P)
    for p in range(P):
        perm=rng.permutation(vals)
        null[p]=perm[nA:].mean()-perm[:nA].mean()
    pv=(np.sum(np.abs(null)>=abs(obs))+1)/(P+1)
    return obs, float(pv)

def _perm_three_region(ed, mB, mT, mA, P=_RP_PERM, rng=None):
    """3 phases Baseline/TOR/Action over [-4,4]. Returns pairwise diffs+p and
    omnibus (variance across the 3 means) + p. Shuffle over union timepoints."""
    rng=rng or _RP_RNG
    idx=np.where(mB|mT|mA)[0]; vals=ed[idx]
    nB=int(mB.sum()); nT=int(mT.sum()); nA=int(mA.sum())
    mb,mt,ma=ed[mB].mean(),ed[mT].mean(),ed[mA].mean()
    obs_pairs={"Baseline-TOR":mb-mt,"TOR-Action":mt-ma,"Baseline-Action":mb-ma}
    obs_omni=np.var([mb,mt,ma])
    nullp={k:np.empty(P) for k in obs_pairs}; null_omni=np.empty(P)
    for p in range(P):
        perm=rng.permutation(vals)
        b=perm[:nB].mean(); t=perm[nB:nB+nT].mean(); a=perm[nB+nT:].mean()
        nullp["Baseline-TOR"][p]=b-t; nullp["TOR-Action"][p]=t-a; nullp["Baseline-Action"][p]=b-a
        null_omni[p]=np.var([b,t,a])
    pvals={k:float((np.sum(np.abs(nullp[k])>=abs(obs_pairs[k]))+1)/(P+1)) for k in obs_pairs}
    p_omni=float((np.sum(null_omni>=obs_omni)+1)/(P+1))   # one-tailed (variance)
    return obs_pairs, pvals, obs_omni, p_omni

def _star(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""

print("="*78)
print("CONTRAST 1: pre vs post (region-label shuffle, two-sided)")
print("="*78)
print(f"{'event':<14}{'mean Pre':>9}{'mean Post':>10}{'Post-Pre':>10}{'p':>8}")
print("-"*78)
_rp1=[]; _ed_for_avg=[]
for _e in _EEVENTS:
    if _e not in _gold_ED_store: continue
    d=_gold_ED_store[_e]; secs=d['secs']; ed=d['ed']; _ed_for_avg.append(ed)
    R=_regions_rp(secs,d['rt'])
    obs,pv=_perm_two_region(ed,R['Pre'],R['Post'])
    print(f"{_EENAME[_e]:<14}{ed[R['Pre']].mean():>9.3f}{ed[R['Post']].mean():>10.3f}{obs:>+10.3f}{pv:>8.3f}{_star(pv)}")
    _rp1.append(dict(event=_EENAME[_e],pre=ed[R['Pre']].mean(),post=ed[R['Post']].mean(),diff=obs,p=pv))
# averaged across events
if len(_ed_for_avg)>1:
    ed_avg=np.mean(_ed_for_avg,0); secs=_gold_ED_store[_EEVENTS[0]]['secs']
    R=_regions_rp(secs,np.nanmean([_gold_ED_store[e]['rt'] for e in _EEVENTS if e in _gold_ED_store]))
    obs,pv=_perm_two_region(ed_avg,R['Pre'],R['Post'])
    print(f"{'AVERAGE':<14}{ed_avg[R['Pre']].mean():>9.3f}{ed_avg[R['Post']].mean():>10.3f}{obs:>+10.3f}{pv:>8.3f}{_star(pv)}")

print("\n"+"="*78)
print("CONTRAST 2: Baseline / TOR / Action (3-phase; pairwise + omnibus)")
print("  boundary = mean RT_Steering per event")
print("="*78)
print(f"{'event':<14}{'Base':>7}{'TOR':>7}{'Action':>7} | {'B-TOR(p)':>14}{'TOR-Act(p)':>15}{'B-Act(p)':>14}{'omni p':>9}")
print("-"*100)
_rp2=[]
for _e in _EEVENTS:
    if _e not in _gold_ED_store: continue
    d=_gold_ED_store[_e]; secs=d['secs']; ed=d['ed']; rt=d['rt']
    R=_regions_rp(secs,rt)
    if R['TOR'].sum()<2 or R['Action'].sum()<2:
        print(f"{_EENAME[_e]:<14} TOR/Action too short (rt={rt:.2f})"); continue
    mb,mt,ma=ed[R['Baseline']].mean(),ed[R['TOR']].mean(),ed[R['Action']].mean()
    pairs,pv,omni,pomni=_perm_three_region(ed,R['Baseline'],R['TOR'],R['Action'])
    print(f"{_EENAME[_e]:<14}{mb:>7.2f}{mt:>7.2f}{ma:>7.2f} | "
          f"{pairs['Baseline-TOR']:>+7.2f}({pv['Baseline-TOR']:.3f}){_star(pv['Baseline-TOR']):<3}"
          f"{pairs['TOR-Action']:>+7.2f}({pv['TOR-Action']:.3f}){_star(pv['TOR-Action']):<3}"
          f"{pairs['Baseline-Action']:>+7.2f}({pv['Baseline-Action']:.3f}){_star(pv['Baseline-Action']):<3}"
          f"{pomni:>9.3f}{_star(pomni)}")
    _rp2.append(dict(event=_EENAME[_e],Baseline=mb,TOR=mt,Action=ma,pairs=pairs,pvals=pv,omni=omni,p_omni=pomni))

print("\nNotes:")
print(f" - {_RP_PERM} permutations, region-label shuffle over timepoints, region sizes fixed.")
print(" - Contrast 1 & pairwise: two-sided. Omnibus: one-tailed (variance across phase means).")
print(" - TOR window = [0, meanRT); short TOR -> few timepoints -> low power. Results in _rp1/_rp2.")
print(" - * p<.05  ** p<.01  *** p<.001.")
'''
ast.parse(code)

nb["cells"].insert(ins+1, {"cell_type":"markdown","metadata":{},"source":md})
nb["cells"].insert(ins+2, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted region-label perm cell after {ins}. Total cells: {len(nb['cells'])}.")
