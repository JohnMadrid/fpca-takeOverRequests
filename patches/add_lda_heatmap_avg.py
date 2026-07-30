import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA_temp_16.6.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
already=any("LDA LOADINGS HEATMAP + AVERAGE CURVE" in ("".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")) for c in nb["cells"])
assert not already, "cell present."

ins=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "TFCE-LDA" in "".join(c["source"]) and "def _run" in "".join(c["source"]): ins=i
assert ins is not None

code = r'''# ============================================================
# LDA LOADINGS HEATMAP + AVERAGE CURVE  (from stored _lda_results, no re-run)
# ------------------------------------------------------------
# (A) Per contrast: heatmap of LD1 variable loadings (cos^2) over time
#     (channel x time). Overlays: TFCE-significant clusters on the x-axis, and
#     a marker band where CV accuracy > 0.60.
# (B) Average accuracy curve across events per contrast (missing from the LDA cell).
# Reads _lda_results (cos2_grid, acc, clusters, label, ch) + _ev[event]["ts"].
# ============================================================
import numpy as np, matplotlib.pyplot as plt

assert "_lda_results" in globals() and "_ev" in globals(), "Run the TFCE-LDA cell first."

_VLAB=["Head.x","Head.y","Eye.x","Eye.y","Steering"]
_NV=len(_VLAB)
_ACC_HI=0.60
def _sm(x,k=5):
    x=np.asarray(x,float); out=x.copy()
    for i in range(len(x)):
        seg=x[max(0,i-k//2):min(len(x),i+k//2+1)]; seg=seg[~np.isnan(seg)]
        if len(seg): out[i]=seg.mean()
    return out
def _skip(info): return "(unbalanced)" in info.get("label","")

# ---------- (A) LD1 loadings heatmap per contrast, per event ----------
for _e in _ev:
    items=[r for r in _lda_results.get(_e,[]) if r is not None and not _skip(r)]
    if not items: continue
    ts=_ev[_e]["ts"]; n=len(items)
    fig,axes=plt.subplots(n,1,figsize=(12,2.6*n),squeeze=False); axes=axes[:,0]
    fig.suptitle(f"{_e} -- LD1 variable loadings (cos²) over time",fontsize=13,fontweight="bold")
    for k,info in enumerate(items):
        ax=axes[k]
        ld1=np.asarray(info["cos2_grid"])[:,0,:].T          # (vars, T): LD1 cos^2
        im=ax.imshow(ld1,aspect="auto",origin="lower",extent=[ts[0],ts[-1],-0.5,_NV-0.5],
                     cmap="magma",vmin=0,vmax=1)
        ax.axvline(0,color="white",ls="--",lw=0.8,alpha=0.7)
        # sig clusters on x-axis (green segments)
        for lo,hi in (info["clusters"] or []):
            x0=ts[lo]; x1=ts[hi-1] if hi-1<len(ts) else ts[-1]
            ax.plot([x0,x1],[-0.5,-0.5],transform=ax.get_xaxis_transform() if False else ax.transData,
                    color="limegreen",lw=6,solid_capstyle="butt",clip_on=False,zorder=6)
        # accuracy>0.60 marker band (top edge)
        acc=_sm(info["acc"]); hi_mask=acc>=_ACC_HI
        ax.scatter(ts[hi_mask], np.full(hi_mask.sum(), _NV-0.5+0.12),
                   marker="s", s=10, color="cyan", clip_on=False, zorder=7)
        ax.set_yticks(range(_NV)); ax.set_yticklabels(_VLAB,fontsize=8)
        ax.set_xlabel("Time (s)"); ax.set_title(f"{info['label']}  (green=TFCE sig; cyan=acc>{_ACC_HI:.0%})",fontsize=9)
        fig.colorbar(im,ax=ax,fraction=0.02,pad=0.01,label="cos²")
    plt.tight_layout(rect=[0,0,1,0.97]); plt.show()

# ---------- (B) average accuracy across events per contrast ----------
_by={}
for _e in _ev:
    for info in _lda_results.get(_e,[]):
        if info is None or _skip(info): continue
        _by.setdefault(info["label"],[]).append((_e,info))
labels=list(_by.keys()); ncol=2; nrow=int(np.ceil(len(labels)/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(7.5*ncol,4.0*nrow),squeeze=False)
fig.suptitle("LDA accuracy averaged across events (per contrast)",fontsize=13,fontweight="bold")
for k,lbl in enumerate(labels):
    rows=_by[lbl]; ts=_ev[rows[0][0]]["ts"]; ch=rows[0][1]["ch"]; col=rows[0][1]["col"]
    accs=np.array([info["acc"] for _e,info in rows]); avg=np.nanmean(accs,0); se=np.nanstd(accs,0,ddof=1)/np.sqrt(len(rows))
    ax=axes[k//ncol][k%ncol]
    ax.plot(ts,_sm(avg),color=col,lw=2.2)
    ax.fill_between(ts,_sm(avg-se),_sm(avg+se),color=col,alpha=0.18,lw=0)
    ax.axhline(ch,color="gray",lw=0.8,ls=":"); ax.axhline(_ACC_HI,color="#888",lw=0.8,ls="--")
    ax.axvline(0,color="k",lw=1.0,ls="--")
    post=ts>=0; pk=np.nanmax(_sm(avg)[post]) if post.any() else np.nan
    ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
    ax.set_title(f"{lbl}  (avg {len(rows)} ev)  post peak={pk:.2f}",fontsize=10); ax.grid(True,alpha=0.25)
for k in range(len(labels),nrow*ncol): axes[k//ncol][k%ncol].axis("off")
plt.tight_layout(rect=[0,0,1,0.96]); plt.show()
print("Heatmap=LD1 cos² loadings; green x-axis=TFCE sig; cyan=acc>60%. Average curves +/-SE across events.")
'''
ast.parse(code)
nb["cells"].insert(ins+1, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code})
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Inserted LDA loadings-heatmap + average-curve cell after {ins}. Total: {len(nb['cells'])}.")
