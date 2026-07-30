import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "LDA PRE/POST + S-V-F BASE/FULL STATS" in "".join(c["source"]): break
s="".join(nb["cells"][i]["source"])

# 1) during the observed pre/post pass, also store (X,y,n_min) per (which,event)
old_store='''        obs,pv,acc=_perm_prepost(Xs,ys,nm,ts)
        pre=ts<0; post=ts>=0
        rows.append((_e,np.nanmean(acc[pre]),np.nanmean(acc[post]),obs,pv))
        _acc_store[(which,_e)]=(ts,acc)'''
new_store='''        obs,pv,acc=_perm_prepost(Xs,ys,nm,ts)
        pre=ts<0; post=ts>=0
        rows.append((_e,np.nanmean(acc[pre]),np.nanmean(acc[post]),obs,pv))
        _acc_store[(which,_e)]=(ts,acc)
        _data_store[(which,_e)]=(Xs,ys,nm)'''
assert old_store in s
s=s.replace(old_store,new_store,1)
s=s.replace("_st1={}; _acc_store={}","_st1={}; _acc_store={}; _data_store={}")

# 2) replace the proxy averaged null with the REAL one (recompute curves under shared flip)
old_block='''    # averaged curve across events for this contrast, then perm on the avg
    accs=[_acc_store[(which,_e)][1] for _e,*_ in rows if (which,_e) in _acc_store]
    ts=_acc_store[(which,rows[0][0])][0]; pre=ts<0; post=ts>=0
    avg=np.nanmean(np.stack(accs),0)
    davg=np.nanmean(avg[post])-np.nanmean(avg[pre])
    # null for averaged curve: reuse per-event nulls is complex; quick proxy = shuffle pre/post of avg timepoints
    rng=np.random.default_rng(1); idx=np.where(pre|post)[0]; nA=int(pre.sum()); nullavg=np.empty(_ST_PERM)
    vals=avg[idx]
    for p in range(_ST_PERM):
        pm=rng.permutation(vals); nullavg[p]=pm[nA:].mean()-pm[:nA].mean()
    pavg=(np.sum(nullavg>=davg)+1)/(_ST_PERM+1)'''
new_block='''    # REAL averaged null: each permutation flips every driver's pre/post (time
    # reversal) in EVERY event, recomputes each event's accuracy curve, averages
    # them, then takes mean(post)-mean(pre) of the averaged curve.
    _evlist=[_e for _e,*_ in rows if (which,_e) in _data_store]
    ts=_acc_store[(which,_evlist[0])][0]; pre=ts<0; post=ts>=0
    avg=np.nanmean(np.stack([_acc_store[(which,_e)][1] for _e in _evlist]),0)
    davg=np.nanmean(avg[post])-np.nanmean(avg[pre])
    rng=np.random.default_rng(1); nullavg=np.empty(_ST_PERM)
    for p in tqdm(range(_ST_PERM),desc=f"avgnull {which}",leave=False):
        accs_p=[]
        for _e in _evlist:
            X,y,nm=_data_store[(which,_e)]
            flip=rng.random(X.shape[0])<0.5
            Xp=X.copy(); Xp[flip,:,:]=Xp[flip][:,::-1,:]
            accp,_=_acc_tc(Xp,y,nm); accs_p.append(accp)
        a=np.nanmean(np.stack(accs_p),0)
        nullavg[p]=np.nanmean(a[post])-np.nanmean(a[pre])
    pavg=(np.sum(nullavg>=davg)+1)/(_ST_PERM+1)'''
assert old_block in s
s=s.replace(old_block,new_block,1)

# update the notes line (remove the proxy caveat)
s=s.replace('print("\\nNotes: (1) one-tailed post>pre, per-driver time-reversal null.")',
            'print("\\nNotes: (1) one-tailed post>pre, per-driver time-reversal null (per-event AND averaged).")')

ast.parse(s)
nb["cells"][i]["source"]=s
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {i}: averaged-curve null is now the REAL time-reversal null (recomputes LDA).")
