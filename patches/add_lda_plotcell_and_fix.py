import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

# ---------- locate LDA cell ----------
lda_i=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "TFCE-LDA (CAR FRAME)" in "".join(c["source"]): lda_i=i; break
assert lda_i is not None

# ========== PART 2: fix the real cell to store null-max arrays ==========
s=nb["cells"][lda_i]["source"]; s="".join(s) if isinstance(s,list) else s
if "null=null" not in s:
    # _run builds 'null' (the TFCE-max null array) but doesn't return it. Add it.
    old="    p=_pval(tfce_obs,null); sig=p<_TFCE_ALPHA\n    return dict(acc=acc,p=p,sig=sig,clusters=_clusters(sig),ch=ch,col=col,n=nt,\n                n_min=n_min,cos2_grid=_cos2_grid(X,y,n_min),label=label)"
    new="    p=_pval(tfce_obs,null); sig=p<_TFCE_ALPHA\n    return dict(acc=acc,p=p,sig=sig,clusters=_clusters(sig),ch=ch,col=col,n=nt,\n                n_min=n_min,cos2_grid=_cos2_grid(X,y,n_min),label=label,\n                tfce_obs=tfce_obs,null=null)"
    assert old in s, "LDA _run return block not found (structure changed?)"
    s=s.replace(old,new,1)
    ast.parse(s); nb["cells"][lda_i]["source"]=s
    fix_msg=f"cell {lda_i}: _run now also stores tfce_obs + null (re-run to populate distributions)."
else:
    fix_msg=f"cell {lda_i}: already stores null; no change."

# ========== PART 1: new plot cell (shaded accuracy time-series only) ==========
already=any("LDA SHADED ACCURACY PLOTS" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "plot cell already present."

md = r'''## LDA accuracy time-series (larger, shaded)

Re-plots the stored car-frame LDA results (_lda_results) as larger shaded
accuracy curves, 2 contrasts per row, plus the average across events per
contrast. Uses existing results -- no re-run of the LDA needed.
'''

code = r'''# ============================================================
# LDA SHADED ACCURACY PLOTS (from stored _lda_results, no re-run)
# ------------------------------------------------------------
# Larger accuracy curves with TFCE-significant clusters shaded, 2 per row.
# Per event + average accuracy across events per contrast.
# ============================================================
import numpy as np, matplotlib.pyplot as plt

def _cf_smooth2(x,k=5):
    x=np.asarray(x,float); out=x.copy()
    for i in range(len(x)):
        seg=x[max(0,i-k//2):min(len(x),i+k//2+1)]; seg=seg[~np.isnan(seg)]
        if len(seg): out[i]=seg.mean()
    return out

def _shade_acc(ax, ts, info):
    ax.plot(ts,_cf_smooth2(info["acc"]),color=info["col"],lw=2.0)
    ax.axhline(info["ch"],color="gray",lw=0.8,ls=":")
    ax.axvline(0,color="k",lw=1.0,ls="--")
    for lo,hi in (info["clusters"] or []):
        ax.axvspan(ts[lo], ts[hi-1] if hi-1<len(ts) else ts[-1], color=info["col"], alpha=0.22)
    ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
    ax.set_title(f"{info['label']}\nn_min/class={info['n_min']}  clusters={len(info['clusters'] or [])}",fontsize=10)
    ax.grid(True,alpha=0.3)

# ---- per event: 2 contrasts per row, large ----
for _e in _ev:
    items=[r for r in _lda_results.get(_e,[]) if r is not None]
    if not items: continue
    ts=_ev[_e]["ts"]; n=len(items); ncol=2; nrow=int(np.ceil(n/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(7.5*ncol,4.2*nrow),squeeze=False)
    fig.suptitle(f"{_e} -- LDA accuracy (car frame)",fontsize=13,fontweight="bold")
    for k,info in enumerate(items):
        _shade_acc(axes[k//ncol][k%ncol], ts, info)
    for k in range(n,nrow*ncol): axes[k//ncol][k%ncol].axis("off")
    plt.tight_layout(rect=[0,0,1,0.96]); plt.show()

# ---- average accuracy across events, per contrast ----
# group stored results by contrast label
_by_label={}
for _e in _ev:
    for info in _lda_results.get(_e,[]):
        if info is None: continue
        _by_label.setdefault(info["label"],[]).append((_e,info))
_labels=list(_by_label.keys())
ncol=2; nrow=int(np.ceil(len(_labels)/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(7.5*ncol,4.2*nrow),squeeze=False)
fig.suptitle("LDA accuracy averaged across events (car frame)",fontsize=13,fontweight="bold")
for k,lbl in enumerate(_labels):
    rows=_by_label[lbl]
    # align on common ts length (events share the same grid)
    accs=np.array([info["acc"] for _e,info in rows])
    ts=_ev[rows[0][0]]["ts"]; ch=rows[0][1]["ch"]; col=rows[0][1]["col"]
    avg=np.nanmean(accs,axis=0)
    ax=axes[k//ncol][k%ncol]
    ax.plot(ts,_cf_smooth2(avg),color=col,lw=2.2)
    ax.axhline(ch,color="gray",lw=0.8,ls=":"); ax.axvline(0,color="k",lw=1.0,ls="--")
    ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
    ax.set_title(f"{lbl}  (avg of {len(rows)} events)",fontsize=10); ax.grid(True,alpha=0.3)
for k in range(len(_labels),nrow*ncol): axes[k//ncol][k%ncol].axis("off")
plt.tight_layout(rect=[0,0,1,0.96]); plt.show()
print("Plotted from stored _lda_results (no LDA re-run). Average = mean accuracy across events per contrast.")
'''
ast.parse(code)

# insert plot cell right after the LDA cell
nb["cells"].insert(lda_i+1, {"cell_type":"markdown","metadata":{},"source":md})
nb["cells"].insert(lda_i+2, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(fix_msg)
print(f"Inserted LDA shaded-accuracy plot cell after {lda_i}. Total cells: {len(nb['cells'])}.")
