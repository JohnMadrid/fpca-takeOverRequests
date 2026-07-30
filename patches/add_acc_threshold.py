import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

# ---------- cell 148: add null-acc-max threshold, peak, frac_above ----------
lda_i=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "TFCE-LDA (CAR FRAME)" in "".join(c["source"]): lda_i=i; break
assert lda_i is not None
s="".join(nb["cells"][lda_i]["source"])

old='''    rng=np.random.default_rng(7); null=np.empty(_TFCE_B)
    for b in _tqdm(range(_TFCE_B),desc=label[:18],leave=False):
        accp,np_=_acc_tc(X,rng.permutation(y),n_min,seed=b+1)
        null[b]=_tfce_1d(np.nan_to_num(_z(accp,ch,np_),nan=0.0)).max()
    p=_pval(tfce_obs,null); sig=p<_TFCE_ALPHA
    return dict(acc=acc,p=p,sig=sig,clusters=_clusters(sig),ch=ch,col=col,n=nt,
                n_min=n_min,cos2_grid=_cos2_grid(X,y,n_min),label=label,
                tfce_obs=tfce_obs,null=null)'''
new='''    rng=np.random.default_rng(7); null=np.empty(_TFCE_B); null_acc_max=np.empty(_TFCE_B)
    for b in _tqdm(range(_TFCE_B),desc=label[:18],leave=False):
        accp,np_=_acc_tc(X,rng.permutation(y),n_min,seed=b+1)
        null[b]=_tfce_1d(np.nan_to_num(_z(accp,ch,np_),nan=0.0)).max()
        null_acc_max[b]=np.nanmax(accp)              # peak accuracy under shuffled labels
    p=_pval(tfce_obs,null); sig=p<_TFCE_ALPHA
    # accuracy threshold = 95th pct of null PEAK accuracy (perm-null-max, MC-aware)
    acc_thresh=float(np.nanpercentile(null_acc_max,95))
    return dict(acc=acc,p=p,sig=sig,clusters=_clusters(sig),ch=ch,col=col,n=nt,
                n_min=n_min,cos2_grid=_cos2_grid(X,y,n_min),label=label,
                tfce_obs=tfce_obs,null=null,null_acc_max=null_acc_max,acc_thresh=acc_thresh)'''
assert old in s, "LDA _run return block (with null) not found"
s=s.replace(old,new,1)
ast.parse(s)
nb["cells"][lda_i]["source"]=s
print(f"cell {lda_i}: _run now stores null_acc_max + acc_thresh (95th pct of null peak accuracy).")

# ---------- plot cell: remove mean lines, add threshold + peak + frac-above ----------
plot_i=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "LDA SHADED ACCURACY PLOTS" in "".join(c["source"]): plot_i=i; break
assert plot_i is not None
ps="".join(nb["cells"][plot_i]["source"])

# replace the per-event _shade_acc mean-line block
old_sh='''    for lo,hi in (info["clusters"] or []):
        ax.axvspan(ts[lo], ts[hi-1] if hi-1<len(ts) else ts[-1], color=info["col"], alpha=0.22)
    _pre=ts<0; _post=ts>=0
    _mpre=np.nanmean(info["acc"][_pre]); _mpost=np.nanmean(info["acc"][_post])
    ax.hlines(_mpre, ts[_pre].min(), 0, color="#444", lw=1.4, ls="-", label=f"pre mean {_mpre:.2f}")
    ax.hlines(_mpost, 0, ts[_post].max(), color="#b22", lw=1.4, ls="-", label=f"post mean {_mpost:.2f}")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0,1)'''
new_sh='''    for lo,hi in (info["clusters"] or []):
        ax.axvspan(ts[lo], ts[hi-1] if hi-1<len(ts) else ts[-1], color=info["col"], alpha=0.22)
    acc=np.asarray(info["acc"]); post=ts>=0
    # threshold: stored null-max (after main-cell re-run) else chance+2SE fallback
    thr=info.get("acc_thresh", np.nan); _thrlab="null-max"
    if not np.isfinite(thr):
        _se=np.sqrt(max(info["ch"]*(1-info["ch"])/max(info.get("n",1),1),1e-12))
        thr=info["ch"]+2*_se; _thrlab="chance+2SE"
    if np.isfinite(thr):
        ax.axhline(thr, color="#555", lw=1.3, ls="--", label=f"{_thrlab} thresh {thr:.2f}")
        above=post & (acc>=thr)
        frac=above[post].mean() if post.any() else np.nan
        ax.fill_between(ts, acc, thr, where=above, color=info["col"], alpha=0.30)
        ax.scatter(ts[above], acc[above], s=10, color=info["col"], zorder=5)
    else:
        frac=np.nan
    pk=np.nanmax(acc[post]) if post.any() else np.nan
    pkt=ts[post][np.nanargmax(acc[post])] if post.any() else np.nan
    if np.isfinite(pk):
        ax.scatter([pkt],[pk], marker="v", s=70, color="k", zorder=6, label=f"peak {pk:.2f}")
    ax.set_title(f"{info['label']}  |  peak={pk:.2f}  above-thresh={frac*100:.0f}% of post",fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0,1)'''
assert old_sh in ps, "per-event mean-line block not found"
ps=ps.replace(old_sh,new_sh,1)

# the _shade_acc set_title earlier sets a 2-line title; our new block also sets title -> remove the earlier one
ps=ps.replace(
'''    ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
    ax.set_title(f"{info['label']}\\nn_min/class={info['n_min']}  clusters={len(info['clusters'] or [])}",fontsize=10)
    ax.grid(True,alpha=0.3)''',
'''    ax.set_ylim(0,1); ax.set_xlabel("Time (s)"); ax.set_ylabel("CV accuracy")
    ax.grid(True,alpha=0.3)''',1)

# average panels: remove mean lines, add peak + threshold (threshold not defined for avg -> use mean of per-event thresholds)
old_avg='''    ax.plot(ts,_cf_smooth2(avg),color=col,lw=2.2)
    ax.axhline(ch,color="gray",lw=0.8,ls=":"); ax.axvline(0,color="k",lw=1.0,ls="--")
    _pre=ts<0; _post=ts>=0
    _mpre=np.nanmean(avg[_pre]); _mpost=np.nanmean(avg[_post])
    ax.hlines(_mpre, ts[_pre].min(), 0, color="#444", lw=1.4, label=f"pre {_mpre:.2f}")
    ax.hlines(_mpost, 0, ts[_post].max(), color="#b22", lw=1.4, label=f"post {_mpost:.2f}")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0,1)'''
new_avg='''    ax.plot(ts,_cf_smooth2(avg),color=col,lw=2.2)
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
    ax.set_ylim(0,1)'''
assert old_avg in ps, "average mean-line block not found"
ps=ps.replace(old_avg,new_avg,1)
# the avg block also had a set_title after; remove the now-duplicate
ps=ps.replace('    ax.set_title(f"{lbl}  (avg of {len(rows)} events)",fontsize=10); ax.grid(True,alpha=0.3)',
              '    ax.grid(True,alpha=0.3)',1)

ast.parse(ps)
nb["cells"][plot_i]["source"]=ps
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {plot_i}: removed mean lines; added null-max threshold, peak marker, above-threshold shading + %.")
