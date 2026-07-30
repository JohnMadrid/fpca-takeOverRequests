import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
already=any("UNBALANCED CONTRASTS (omitted earlier)" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "cell present."
# insert just BEFORE the TFCE 1-dim cell
ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "TFCE ON 1-DIM (PC1) LDA" in "".join(c["source"]): ins=i
assert ins is not None

code = r'''# ============================================================
# UNBALANCED CONTRASTS (omitted earlier) -- PCA-LDA 1/2/5 dims, no TFCE
# ------------------------------------------------------------
# The unbalanced designs (Base vs Full all drivers; Success vs Failure all
# drivers) that we left out of the main plots. Same gold PCA -> LDA on top
# 1/2/5 PCs, CV accuracy, no permutation test. For reference only (unbalanced =
# confound with the other factor not controlled).
# Reuses _ev, _design, _pd_acc_curve, _pd_smooth, _PD_DIMS/_DCOL/_DSTY.
# ============================================================
import numpy as np, matplotlib.pyplot as plt

_UB_DESIGNS=["BvF_unbal","SvF_unbal"]
for _e in _ev:
    ts=_ev[_e]["ts"]
    for which in _UB_DESIGNS:
        des=_design(_ev[_e],which)
        if des is None: continue
        lbl,Xs,ys,nm,ch=des
        if nm<3: print(f"{_e} | {lbl}: low n"); continue
        fig,ax=plt.subplots(figsize=(9,4))
        for kd in _PD_DIMS:
            a,_=_pd_acc_curve(Xs,ys,nm,kd)
            ax.plot(ts,_pd_smooth(a),color=_PD_DCOL[kd],ls=_PD_DSTY[kd],lw=1.3,label=f"{kd}-dim (top {kd} PC)")
        ax.axhline(ch,color="gray",lw=0.8,ls=":"); ax.axvline(0,color="k",lw=1.0,ls="--")
        ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
        ax.set_title(f"{_e} -- {lbl}  (UNBALANCED; reference only)",fontsize=11)
        ax.grid(True,alpha=0.25); ax.legend(fontsize=8,loc="lower right")
        plt.tight_layout(); plt.show()
print("Unbalanced contrasts shown for reference (the other factor is not balanced out).")
'''
ast.parse(code)
nb["cells"].insert(ins, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted unbalanced-curves cell at {ins} (above the TFCE 1-dim cell). Total: {len(nb['cells'])}.")
