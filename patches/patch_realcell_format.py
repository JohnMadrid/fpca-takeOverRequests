import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "TFCE-LDA (CAR FRAME)" in "".join(c["source"]): break
s="".join(nb["cells"][i]["source"])

# 1) drop unbalanced designs
s=s.replace('_DESIGNS = ["BvF_unbal","BvF_bal","SvF_unbal","SvF_bal","SvF_base","SvF_full"]',
            '_DESIGNS = ["BvF_bal","SvF_bal","SvF_base","SvF_full"]')
s=s.replace('''_DCOL = {"BvF_unbal":"#9467bd","BvF_bal":"#6a3d9a","SvF_unbal":"#d62728","SvF_bal":"#a01619",
         "SvF_base":"#1f77b4","SvF_full":"#ff7f0e"}''',
            '''_DCOL = {"BvF_bal":"#6a3d9a","SvF_bal":"#a01619","SvF_base":"#1f77b4","SvF_full":"#ff7f0e"}''')

# 2) add gradient + x-axis cluster helpers before def _plot
anchor='def _plot(results, ts, title):'
helpers='''def _grad_fill_lda(ax, ts, acc, ch):
    """Fixed vertical diverging gradient (blue<chance, white=chance, red>chance),
    revealed only between the accuracy curve and the chance line."""
    import matplotlib as _mpl
    acc=np.asarray(acc); ts=np.asarray(ts)
    lo=min(ch,float(np.nanmin(acc))); hi=max(float(np.nanmax(acc)),ch)
    ny=200; yg=np.linspace(lo,hi,ny); half=max(hi-ch,ch-lo,1e-6)
    rowcol=_mpl.colormaps['RdBu_r']((np.clip((yg-ch)/half,-1,1)+1)/2)
    rgba=np.repeat(rowcol[:,None,:],len(ts),axis=1)
    A=np.zeros((ny,len(ts)))
    for j in range(len(ts)):
        a=acc[j]
        if np.isnan(a): continue
        y0,y1=min(ch,a),max(ch,a); A[(yg>=y0)&(yg<=y1),j]=1.0
    rgba[...,3]=A*0.85
    ax.imshow(rgba,aspect='auto',origin='lower',extent=[ts[0],ts[-1],lo,hi],
              zorder=1,interpolation='bilinear')

def _xaxis_clusters_lda(ax, ts, clusters, color="limegreen"):
    if not clusters: return
    for lo,hi in clusters:
        x0=ts[lo]; x1=ts[hi-1] if hi-1<len(ts) else ts[-1]
        ax.plot([x0,x1],[0.012,0.012],transform=ax.get_xaxis_transform(),
                color=color,lw=5,solid_capstyle="butt",zorder=6,clip_on=False)

def _plot(results, ts, title):'''
assert anchor in s
s=s.replace(anchor,helpers,1)

# 3) rewrite the accuracy panel (axes[0,ci]) to the gradient + x-axis format
old_acc='''        ax=axes[0,ci]
        ax.plot(ts,_cf_smooth(info["acc"]),color=info["col"],lw=1.6)
        ax.axhline(info["ch"],color="gray",lw=0.6,ls=":"); ax.axvline(0,color="k",lw=0.8,ls="--")
        for lo,hi in (info["clusters"] or []):
            ax.axvspan(ts[lo],ts[hi-1] if hi-1<len(ts) else ts[-1],color=info["col"],alpha=0.20)
        ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
        ax.set_title(f"{info['label']}\\nn_min/class={info['n_min']}  clusters={len(info['clusters'] or [])}",fontsize=8)'''
new_acc='''        ax=axes[0,ci]
        _acc=_cf_smooth(info["acc"]); _ch=info["ch"]; _post=ts>=0
        _grad_fill_lda(ax, ts, _acc, _ch)
        ax.plot(ts,_acc,color="#222",lw=1.8,zorder=4)
        ax.axhline(_ch,color="gray",lw=0.8,ls=":",zorder=3); ax.axvline(0,color="k",lw=1.0,ls="--",zorder=3)
        _xaxis_clusters_lda(ax, ts, info["clusters"] or [])
        _pk=np.nanmax(_acc[_post]) if _post.any() else np.nan
        if np.isfinite(_pk):
            ax.hlines(_pk,0.0,ts[_post].max(),color="k",lw=1.2,alpha=0.4,label=f"post max {_pk:.2f}")
            ax.legend(fontsize=8,loc="lower right")
        ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
        ax.set_title(f"{info['label']}  n/class={info['n_min']}  post peak={_pk:.2f}",fontsize=9)
        ax.grid(True,alpha=0.15,zorder=0)'''
assert old_acc in s
s=s.replace(old_acc,new_acc,1)

ast.parse(s)
nb["cells"][i]["source"]=s
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {i}: gradient+x-axis accuracy format applied; unbalanced designs removed ({_DESIGNS if False else '4 designs'}).")
