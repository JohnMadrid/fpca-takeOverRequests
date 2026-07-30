import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "LDA SHADED ACCURACY PLOTS" in "".join(c["source"]): break
s="".join(nb["cells"][i]["source"])

# helper to draw clusters as x-axis bars (added once near top of cell, after _cf_smooth2)
helper = '''def _cf_smooth2(x,k=5):'''
xaxis_fn = '''def _xaxis_clusters(ax, ts, clusters, color="limegreen"):
    """Mark TFCE-significant clusters as thick segments along the bottom x-axis."""
    if not clusters: return
    y=0.012  # just above the axis in axes-fraction
    for lo,hi in clusters:
        x0=ts[lo]; x1=ts[hi-1] if hi-1<len(ts) else ts[-1]
        ax.plot([x0,x1],[y,y],transform=ax.get_xaxis_transform(),
                color=color,lw=5,solid_capstyle="butt",zorder=6,clip_on=False)

def _cf_smooth2(x,k=5):'''
assert helper in s
s=s.replace(helper, xaxis_fn, 1)

# per-event: remove axvspan, add x-axis clusters
old_av='''    for lo,hi in (info["clusters"] or []):
        ax.axvspan(ts[lo], ts[hi-1] if hi-1<len(ts) else ts[-1], color="k", alpha=0.07, zorder=2)'''
new_av='''    _xaxis_clusters(ax, ts, info["clusters"] or [])'''
assert old_av in s
s=s.replace(old_av,new_av,1)

# averaged panel: replace whole plotting body with gradient + x-axis clusters
old_avg_block='''    ax.plot(ts,_cf_smooth2(avg),color=col,lw=2.2)
    ax.axhline(ch,color="gray",lw=0.8,ls=":"); ax.axvline(0,color="k",lw=1.0,ls="--")
    def _thr_of(info):
        t=info.get("acc_thresh",np.nan)
        if np.isfinite(t): return t
        _se=np.sqrt(max(info["ch"]*(1-info["ch"])/max(info.get("n",1),1),1e-12))
        return info["ch"]+2*_se
    _thr=np.nanmean([_thr_of(info) for _e,info in rows])
    post=ts>=0
    if np.isfinite(_thr):
        ax.axhline(_thr,color="#555",lw=1.3,ls="--",label=f"mean thresh {_thr:.2f}")
        above=post & (avg>=_thr); frac=above[post].mean() if post.any() else np.nan
        ax.fill_between(ts,avg,_thr,where=above,color=col,alpha=0.30)
    else: frac=np.nan
    pk=np.nanmax(avg[post]) if post.any() else np.nan
    ax.set_title(f"{lbl}  (avg {len(rows)} ev)  peak={pk:.2f}  above={frac*100:.0f}%",fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
    ax.grid(True,alpha=0.3)'''
new_avg_block='''    avg=_cf_smooth2(avg)
    _grad_fill(ax, ts, avg, ch)
    ax.plot(ts,avg,color="#222",lw=1.8,zorder=4)
    ax.axhline(ch,color="gray",lw=0.8,ls=":",zorder=3); ax.axvline(0,color="k",lw=1.0,ls="--",zorder=3)
    # union of significant clusters across events, on the x-axis
    _allcl=[]
    for _e,info in rows: _allcl += (info["clusters"] or [])
    _xaxis_clusters(ax, ts, _allcl)
    post=ts>=0; pk=np.nanmax(avg[post]) if post.any() else np.nan
    if np.isfinite(pk):
        ax.hlines(pk,0.0,ts[post].max(),color="k",lw=1.2,alpha=0.4,label=f"post max {pk:.2f}")
    ax.set_title(f"{lbl}  (avg {len(rows)} ev)  post peak={pk:.2f}",fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
    ax.grid(True,alpha=0.15,zorder=0)'''
assert old_avg_block in s
s=s.replace(old_avg_block,new_avg_block,1)

ast.parse(s)
nb["cells"][i]["source"]=s
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {i}: clusters on x-axis (per-event + averaged); averaged panel now uses gradient.")
