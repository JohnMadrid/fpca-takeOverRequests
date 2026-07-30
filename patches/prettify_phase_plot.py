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
# Values from the GOLD region-perm cell (_rp2 + _gold_ED_store). Phases placed at
# their real time-window centres on a -4..4 axis (approx; category labels kept).
# Run the gold ED + REGION-LABEL PERMUTATION TEST cells first.
# ============================================================
import numpy as np, matplotlib.pyplot as plt

assert "_rp2" in globals() and _rp2, "Run the gold REGION-LABEL PERMUTATION TEST cell first (_rp2 missing)."
assert "_gold_ED_store" in globals(), "Run the gold ED cell first (_gold_ED_store missing)."

# muted, professional palette (events)
_PCOL={"Stag crossing":"#4C72B0","Falling rocks":"#55A868","Motorcyclist":"#C44E52"}
_AVG_COL="#7f7f7f"
def _st(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"

# x positions = phase-window centres on the real -4..4 axis, using mean RT
_evk=[e for e in _EEVENTS if e in _gold_ED_store]
_rt_mean=float(np.nanmean([_gold_ED_store[e]["rt"] for e in _evk]))
xpos=[-2.0, _rt_mean/2.0, (_rt_mean+4.0)/2.0]          # Baseline, TOR, Action centres
xlab=[f"Baseline\n[-4, 0)", f"TOR\n[0, {_rt_mean:.1f})", f"Action\n[{_rt_mean:.1f}, 4]"]

plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,
                     "font.size":11,"axes.titlesize":12,"axes.labelsize":11})
fig,ax=plt.subplots(figsize=(8.2,5.4))

# faint phase backgrounds on the real time axis
ax.axvspan(-4,0,color="#f2f2f2",zorder=0)
ax.axvspan(0,_rt_mean,color="#eaeaf2",zorder=0)
ax.axvspan(_rt_mean,4,color="#f2eaea",zorder=0)
ax.axvline(0,color="#bbbbbb",lw=0.8,ls="--",zorder=1)
ax.axvline(_rt_mean,color="#bbbbbb",lw=0.8,ls=":",zorder=1)

# --- per event ---
for r in _rp2:
    name=r["event"]; m=[r["Baseline"],r["TOR"],r["Action"]]; col=_PCOL.get(name,"#555")
    ax.plot(xpos,m,'o-',color=col,lw=1.8,ms=7,label=name,zorder=4,markeredgecolor="white",markeredgewidth=0.8)
    for (xa,xb,ya,yb,p) in [(xpos[0],xpos[1],m[0],m[1],r["pvals"]["Baseline-TOR"]),
                             (xpos[1],xpos[2],m[1],m[2],r["pvals"]["TOR-Action"])]:
        ax.annotate(_st(p),((xa+xb)/2,(ya+yb)/2),fontsize=8.5,color=col,ha="center",va="bottom")

# --- average (thin gray) ---
if len(_evk)>1:
    secs=_gold_ED_store[_evk[0]]["secs"]
    edavg=np.mean([_gold_ED_store[e]["ed"] for e in _evk],0)
    Ra=_regions_rp(secs,_rt_mean)
    mavg=[edavg[Ra["Baseline"]].mean(),edavg[Ra["TOR"]].mean(),edavg[Ra["Action"]].mean()]
    ax.plot(xpos,mavg,'o-',color=_AVG_COL,lw=2.0,ms=7,label="Average",zorder=5,
            markeredgecolor="white",markeredgewidth=0.8)
    _,pbt=_perm_two_region(edavg,Ra["TOR"],Ra["Baseline"])
    _,pta=_perm_two_region(edavg,Ra["Action"],Ra["TOR"])
    ax.annotate(_st(pbt),((xpos[0]+xpos[1])/2,(mavg[0]+mavg[1])/2),fontsize=9,color="#444",ha="center",va="top",fontweight="bold")
    ax.annotate(_st(pta),((xpos[1]+xpos[2])/2,(mavg[1]+mavg[2])/2),fontsize=9,color="#444",ha="center",va="top",fontweight="bold")

ax.set_xlim(-4,4); ax.set_xticks(xpos); ax.set_xticklabels(xlab,fontsize=9.5)
# secondary tick marks for the true time axis
ax2=ax.secondary_xaxis("top"); ax2.set_xticks([-4,-2,0,2,4]); ax2.set_xlabel("Time from onset (s)",fontsize=9)
ax.set_ylabel("Effective dimensionality (gold ED)")
ax.set_title("ED across phases: Baseline / Take-over request / Action")
ax.grid(True,axis="y",alpha=0.25,zorder=0)
ax.legend(frameon=False,fontsize=9.5,loc="best")
plt.tight_layout(); plt.show()
print("Stars: *** p<.001  ** p<.01  * p<.05  ns. Phases placed at real time-window centres (mean RT boundary).")
'''
ast.parse(code)
nb["cells"][pi]["source"]=code
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {pi}: prettified phase plot (real time axis, gray average, muted palette).")
