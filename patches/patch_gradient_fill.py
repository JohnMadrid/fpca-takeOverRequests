import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "LDA SHADED ACCURACY PLOTS" in "".join(c["source"]): break
s="".join(nb["cells"][i]["source"])

old = '''def _shade_acc(ax, ts, info):
    acc=_cf_smooth2(np.asarray(info["acc"]))   # single smoothed curve used everywhere
    ax.plot(ts,acc,color=info["col"],lw=2.0)
    ax.axhline(info["ch"],color="gray",lw=0.8,ls=":")
    ax.axvline(0,color="k",lw=1.0,ls="--")
    for lo,hi in (info["clusters"] or []):
        ax.axvspan(ts[lo], ts[hi-1] if hi-1<len(ts) else ts[-1], color=info["col"], alpha=0.22)
    post=ts>=0
    # threshold: stored null-max (after main-cell re-run) else chance+2SE fallback
    thr=info.get("acc_thresh", np.nan); _thrlab="null-max"
    if not np.isfinite(thr):
        _se=np.sqrt(max(info["ch"]*(1-info["ch"])/max(info.get("n",1),1),1e-12))
        thr=info["ch"]+2*_se; _thrlab="chance+2SE"
    _t0,_t1=0.0,ts[post].max() if post.any() else 0.0
    if np.isfinite(thr):
        ax.hlines(thr,_t0,_t1, color="#aaaaaa", lw=1.2, ls="--", label=f"{_thrlab} thresh {thr:.2f}")
        above=post & (acc>=thr)
        frac=above[post].mean() if post.any() else np.nan
        ax.fill_between(ts, acc, thr, where=above, color=info["col"], alpha=0.30)
    else:
        frac=np.nan
    pk=np.nanmax(acc[post]) if post.any() else np.nan
    if np.isfinite(pk):
        ax.hlines(pk,_t0,_t1, color="k", lw=1.6, label=f"post max {pk:.2f}")
    ax.set_title(f"{info['label']}  |  peak={pk:.2f}  above-thresh={frac*100:.0f}% of post",fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
    ax.grid(True,alpha=0.3)'''

new = '''def _grad_fill(ax, ts, acc, ch, vmax=None, cmap="RdBu_r"):
    """Vertical blue->red gradient between the chance line and the accuracy curve.
    Colour encodes accuracy: near chance = blue, high accuracy = red. Only the
    band between chance and acc is shown (rest masked)."""
    import numpy as _np
    acc=_np.asarray(acc); ts=_np.asarray(ts)
    lo=min(ch, float(_np.nanmin(acc))); hi=max(float(_np.nanmax(acc)), ch)
    ny=240
    ygrid=_np.linspace(lo,hi,ny)
    # colour value = the y position itself (accuracy), so colour tracks height
    img=_np.repeat(ygrid[:,None], len(ts), axis=1)            # (ny, T) value=accuracy-at-row
    # mask: keep only cells between chance and the accuracy curve (either side)
    mask=_np.zeros_like(img,bool)
    for j in range(len(ts)):
        a=acc[j]
        if _np.isnan(a): continue
        y0,y1=min(ch,a),max(ch,a)
        mask[:,j]=(ygrid>=y0)&(ygrid<=y1)
    arr=_np.ma.masked_where(~mask, img)
    if vmax is None: vmax=hi
    vmin=2*ch-vmax    # symmetric about chance so chance maps to white centre
    ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap,
              extent=[ts[0],ts[-1],lo,hi], vmin=vmin, vmax=vmax, zorder=1)

def _shade_acc(ax, ts, info):
    acc=_cf_smooth2(np.asarray(info["acc"]))   # single smoothed curve used everywhere
    ch=info["ch"]; post=ts>=0
    # blue->red gradient between chance and accuracy, colour = accuracy
    _grad_fill(ax, ts, acc, ch)
    ax.plot(ts,acc,color="k",lw=1.6,zorder=4)
    ax.axhline(ch,color="gray",lw=0.8,ls=":",zorder=3)
    ax.axvline(0,color="k",lw=1.0,ls="--",zorder=3)
    for lo,hi in (info["clusters"] or []):
        ax.axvspan(ts[lo], ts[hi-1] if hi-1<len(ts) else ts[-1], color="k", alpha=0.07, zorder=2)
    _t0,_t1=0.0,(ts[post].max() if post.any() else 0.0)
    pk=np.nanmax(acc[post]) if post.any() else np.nan
    if np.isfinite(pk):
        ax.hlines(pk,_t0,_t1, color="k", lw=1.2, alpha=0.4, label=f"post max {pk:.2f}")
    ax.set_title(f"{info['label']}  |  post peak={pk:.2f}",fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
    ax.grid(True,alpha=0.15,zorder=0)'''

assert old in s, "shade_acc block not found"
s=s.replace(old,new,1)
ast.parse(s)
nb["cells"][i]["source"]=s
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {i}: gradient fill added; threshold line removed; max line faint.")
