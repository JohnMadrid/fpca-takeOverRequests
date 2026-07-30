import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "LDA ON PCA COMPONENTS" in "".join(c["source"]): break
s="".join(nb["cells"][i]["source"])

# 1) init accumulator + ref ts before the event loop
anchor='# ---- per event, per contrast: one fused figure (accuracy 1/2/5 + PC1 heatmap) ----\nfor _e in _ev:'
acc_init=('# accumulate across events for the average panel\n'
          '_PD_ACCUM={w:{kd:[] for kd in _PD_DIMS} for w in _PD_DESIGNS}\n'
          '_PD_PC1ACC={w:[] for w in _PD_DESIGNS}\n'
          '_PD_REF_TS=_ev[list(_ev)[0]]["ts"]\n'
          '# ---- per event, per contrast: one fused figure (accuracy 1/2/5 + PC1 heatmap) ----\n'
          'for _e in _ev:')
assert anchor in s
s=s.replace(anchor,acc_init,1)

# 2) inside the loop, after accs computed, store them
old_store='''        accs={}; pc1grid=None
        for kd in _PD_DIMS:
            a,g=_pd_acc_curve(Xs,ys,nm,kd); accs[kd]=a
            if kd==1: pc1grid=g'''
new_store='''        accs={}; pc1grid=None
        for kd in _PD_DIMS:
            a,g=_pd_acc_curve(Xs,ys,nm,kd); accs[kd]=a
            if kd==1: pc1grid=g
        _tsE=_ev[_e]["ts"]
        def _aln(c): return c if (len(_tsE)==len(_PD_REF_TS) and np.allclose(_tsE,_PD_REF_TS)) else np.interp(_PD_REF_TS,_tsE,c,left=np.nan,right=np.nan)
        for kd in _PD_DIMS: _PD_ACCUM[which][kd].append(_aln(accs[kd]))
        _PD_PC1ACC[which].append(np.array([_aln(pc1grid[m]) for m in range(_PD_M)]))'''
assert old_store in s
s=s.replace(old_store,new_store,1)

# 3) after the loop (before the final print), add the average figures
final_print='print("Accuracy: LDA on top-1/2/5 PCA components per timepoint'
avg_block='''# ---- AVERAGE across events per contrast (same fused layout) ----
_ts=_PD_REF_TS
for which in _PD_DESIGNS:
    if not _PD_ACCUM[which][_PD_DIMS[0]]: continue
    fig,(axA,axH)=plt.subplots(2,1,figsize=(9,5.6),sharex=True,
                               gridspec_kw={"height_ratios":[2.3,1.0],"hspace":0.08})
    _nE=len(_PD_ACCUM[which][_PD_DIMS[0]])
    for kd in _PD_DIMS:
        arr=np.array(_PD_ACCUM[which][kd]); m=np.nanmean(arr,0)
        axA.plot(_ts,_pd_smooth(m),color=_PD_DCOL[kd],ls=_PD_DSTY[kd],lw=1.3,label=f"{kd}-dim (top {kd} PC)")
    axA.axhline(0.5,color="gray",lw=0.8,ls=":"); axA.axvline(0,color="k",lw=1.0,ls="--")
    axA.set_ylim(0,1); axA.set_ylabel("CV accuracy"); axA.legend(fontsize=8,loc="lower right")
    axA.set_title(f"AVERAGE ({_nE} events) -- {which}   (LDA on top PCA dims)",fontsize=11); axA.grid(True,alpha=0.25)
    _dA=make_axes_locatable(axA); _cA=_dA.append_axes("right",size="2.5%",pad=0.08); _cA.axis("off")
    pc1avg=np.nanmean(np.array(_PD_PC1ACC[which]),0)   # (M,T)
    im=axH.imshow(pc1avg,aspect="auto",origin="lower",extent=[_ts[0],_ts[-1],-0.5,_PD_M-0.5],cmap="magma",vmin=0,vmax=1)
    axH.axvline(0,color="white",ls="--",lw=0.8,alpha=0.7)
    axH.set_yticks(range(_PD_M)); axH.set_yticklabels(_PD_VLAB,fontsize=8)
    axH.set_xlabel("Time (s)"); axH.set_ylabel("PC1 cos²",fontsize=9)
    _dH=make_axes_locatable(axH); _cH=_dH.append_axes("right",size="2.5%",pad=0.08); fig.colorbar(im,cax=_cH,label="cos²")
    plt.tight_layout(); plt.show()

'''+final_print
assert final_print in s
s=s.replace(final_print,avg_block,1)

ast.parse(s)
nb["cells"][i]["source"]=s
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {i}: added across-event AVERAGE figures (1/2/5-dim curves + averaged PC1 heatmap) per contrast.")
