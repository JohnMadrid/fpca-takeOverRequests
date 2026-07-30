import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

# locate plot cell + separate heatmap cell
pi=hi=None
for i,c in enumerate(nb["cells"]):
    s="".join(c["source"]) if isinstance(c["source"],list) else c.get("source","")
    if c["cell_type"]=="code" and "LDA SHADED ACCURACY PLOTS" in s: pi=i
    if c["cell_type"]=="code" and "LDA LOADINGS HEATMAP + AVERAGE CURVE" in s: hi=i
assert pi is not None

s="".join(nb["cells"][pi]["source"])

# helper to draw the LD1 heatmap on a given axis (added after _shade_acc def)
heat_helper = '''def _ld1_heatmap(ax, ts, info, vlab):
    """LD1 variable loadings (cos^2) over time: channel x time heatmap, matched x."""
    import numpy as _np
    nv=len(vlab)
    ld1=_np.asarray(info["cos2_grid"])[:,0,:].T          # (vars, T)
    im=ax.imshow(ld1,aspect="auto",origin="lower",extent=[ts[0],ts[-1],-0.5,nv-0.5],
                 cmap="magma",vmin=0,vmax=1)
    ax.axvline(0,color="white",ls="--",lw=0.8,alpha=0.7)
    for lo,hi in (info["clusters"] or []):
        x0=ts[lo]; x1=ts[hi-1] if hi-1<len(ts) else ts[-1]
        ax.plot([x0,x1],[nv-0.5+0.18,nv-0.5+0.18],color="limegreen",lw=5,
                solid_capstyle="butt",clip_on=False,zorder=6)
    ax.set_yticks(range(nv)); ax.set_yticklabels(vlab,fontsize=8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("LD1 cos²",fontsize=9)
    return im

'''
# insert heat helper right before the "# ---- per event" block
anchor='# ---- per event: 2 contrasts per row, large ----'
assert anchor in s
s=s.replace(anchor, heat_helper+anchor, 1)

# replace the per-event block: one fused figure (acc + heatmap) PER CONTRAST
old_block='''# ---- per event: 2 contrasts per row, large ----
for _e in _ev:
    items=[r for r in _lda_results.get(_e,[]) if r is not None and not _SKIP_UNBAL(r)]
    if not items: continue
    ts=_ev[_e]["ts"]; n=len(items); ncol=2; nrow=int(np.ceil(n/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(7.5*ncol,4.2*nrow),squeeze=False)
    fig.suptitle(f"{_e} -- LDA accuracy (car frame)",fontsize=13,fontweight="bold")
    for k,info in enumerate(items):
        _shade_acc(axes[k//ncol][k%ncol], ts, info)
    for k in range(n,nrow*ncol): axes[k//ncol][k%ncol].axis("off")
    plt.tight_layout(rect=[0,0,1,0.96]); plt.show()'''
new_block='''_VLAB_HEAT=["Head.x","Head.y","Eye.x","Eye.y","Steering"]
# ---- per event: one fused figure per contrast (accuracy on top, LD1 heatmap below, shared x) ----
for _e in _ev:
    items=[r for r in _lda_results.get(_e,[]) if r is not None and not _SKIP_UNBAL(r)]
    if not items: continue
    ts=_ev[_e]["ts"]
    for info in items:
        fig,(axA,axH)=plt.subplots(2,1,figsize=(9,5.6),sharex=True,
                                   gridspec_kw={"height_ratios":[2.3,1.0],"hspace":0.08})
        _shade_acc(axA, ts, info)
        axA.set_xlabel("")
        im=_ld1_heatmap(axH, ts, info, _VLAB_HEAT)
        fig.colorbar(im,ax=axH,fraction=0.03,pad=0.01,label="cos²")
        fig.suptitle(f"{_e} -- {info['label']}",fontsize=12,fontweight="bold")
        plt.tight_layout(rect=[0,0,1,0.97]); plt.show()'''
assert old_block in s
s=s.replace(old_block,new_block,1)

ast.parse(s)
nb["cells"][pi]["source"]=s

# remove the now-redundant separate heatmap part: keep only the AVERAGE-curve part of cell hi
if hi is not None:
    sh="".join(nb["cells"][hi]["source"])
    # the separate cell did (A) heatmaps + (B) average. Heatmaps now fused -> strip (A), keep (B).
    # simplest: delete the heatmap cell entirely IF the plot cell now also lacks the average.
    # The plot cell (pi) already has its own average block, so the separate cell is fully redundant -> delete.
    del nb["cells"][hi]
    note=f"deleted redundant separate heatmap cell {hi}"
else:
    note="no separate heatmap cell found"

json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"plot cell {pi}: accuracy+LD1 heatmap fused per contrast (shared x). {note}. Total: {len(nb['cells'])}.")
