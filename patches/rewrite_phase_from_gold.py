import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA_temp_16.6.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

pi=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "PHASE-MEAN PLOT" in "".join(c["source"]): pi=i; break
assert pi is not None

code = r'''# ============================================================
# PHASE-MEAN PLOT (Baseline/TOR/Action) -- 3 events + average, one figure
# ------------------------------------------------------------
# Reads values from the GOLD region-perm cell:
#   _rp2          -> per-event Baseline/TOR/Action means + pairwise p's
#   _gold_ED_store-> ED curves + rt (for the averaged line + its stars)
# Gold ED (C + pooled scale). Run the "REGION-LABEL PERMUTATION TEST (gold ED)"
# cell first. x = 3 phases; coloured line per event; thick black = average.
# Stars on Base->TOR and TOR->Action from the pairwise region-shuffle p.
# ============================================================
import numpy as np, matplotlib.pyplot as plt

assert "_rp2" in globals() and _rp2, "Run the gold REGION-LABEL PERMUTATION TEST cell first (_rp2 missing)."
assert "_gold_ED_store" in globals(), "Run the gold ED cell first (_gold_ED_store missing)."

_PCOL={"Stag crossing":"steelblue","Falling rocks":"#2ca02c","Motorcyclist":"#d62728"}
xpos=[0,1,2]; xlab=["Baseline","TOR","Action"]
def _st(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"

fig,ax=plt.subplots(figsize=(8.5,6))
# --- per event from _rp2 ---
for r in _rp2:
    name=r["event"]; m=[r["Baseline"],r["TOR"],r["Action"]]
    col=_PCOL.get(name,"gray")
    ax.plot(xpos,m,'o-',color=col,lw=2,ms=8,label=name)
    pbt=r["pvals"]["Baseline-TOR"]; pta=r["pvals"]["TOR-Action"]
    ax.annotate(_st(pbt),((xpos[0]+xpos[1])/2,(m[0]+m[1])/2),fontsize=10,color=col,ha="center",va="bottom",fontweight="bold")
    ax.annotate(_st(pta),((xpos[1]+xpos[2])/2,(m[1]+m[2])/2),fontsize=10,color=col,ha="center",va="bottom",fontweight="bold")

# --- average line from _gold_ED_store (averaged ED curve, same recipe) ---
_evk=[e for e in _EEVENTS if e in _gold_ED_store]
if len(_evk)>1:
    secs=_gold_ED_store[_evk[0]]["secs"]
    edavg=np.mean([_gold_ED_store[e]["ed"] for e in _evk],0)
    rt_avg=np.nanmean([_gold_ED_store[e]["rt"] for e in _evk])
    Ra=_regions_rp(secs,rt_avg)
    mavg=[edavg[Ra["Baseline"]].mean(),edavg[Ra["TOR"]].mean(),edavg[Ra["Action"]].mean()]
    ax.plot(xpos,mavg,'o-',color="k",lw=3.2,ms=9,label="Average",zorder=5)
    # average stars via the same pairwise region-shuffle on the averaged curve
    _,pbt=_perm_two_region(edavg,Ra["TOR"],Ra["Baseline"])   # TOR vs Baseline
    _,pta=_perm_two_region(edavg,Ra["Action"],Ra["TOR"])     # Action vs TOR
    ax.annotate(_st(pbt),((xpos[0]+xpos[1])/2,(mavg[0]+mavg[1])/2),fontsize=11,color="k",ha="center",va="top",fontweight="bold")
    ax.annotate(_st(pta),((xpos[1]+xpos[2])/2,(mavg[1]+mavg[2])/2),fontsize=11,color="k",ha="center",va="top",fontweight="bold")

ax.set_xticks(xpos); ax.set_xticklabels(xlab,fontsize=11)
ax.set_ylabel("Effective dimensionality (gold ED)",fontsize=11)
ax.set_title("ED by phase: Baseline / Take-over request / Action\n(stars = pairwise region-shuffle p, gold ED)",fontsize=12,fontweight="bold")
ax.grid(True,alpha=0.3); ax.legend(fontsize=10)
plt.tight_layout(); plt.show()
print("Stars: *** p<.001  ** p<.01  * p<.05  ns. Values from _rp2 / _gold_ED_store (gold recipe).")
'''
ast.parse(code)
nb["cells"][pi]["source"]=code
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {pi}: phase plot now reads from _rp2 + _gold_ED_store (gold).")
