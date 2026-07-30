import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA_temp_16.6.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "PHASE-MEAN PLOT" in "".join(c["source"]): break
s="".join(nb["cells"][i]["source"])

# 1) Okabe-Ito event palette
old_pal='_PCOL={"Stag crossing":"#4C72B0","Falling rocks":"#55A868","Motorcyclist":"#C44E52"}'
new_pal='_PCOL={"Stag crossing":"#0072B2","Falling rocks":"#009E73","Motorcyclist":"#D55E00"}  # Okabe-Ito'
assert old_pal in s, "palette line not found"
s=s.replace(old_pal,new_pal,1)

# 2) average block -> black dashed + diamonds + SE ribbon
old_avg='''    ax.plot(xpos,mavg,'o-',color=_AVG_COL,lw=2.0,ms=7,label="Average",zorder=5,
            markeredgecolor="white",markeredgewidth=0.8)
    _,pbt=_perm_two_region(edavg,Ra["TOR"],Ra["Baseline"])
    _,pta=_perm_two_region(edavg,Ra["Action"],Ra["TOR"])'''
new_avg='''    # +/-SE ribbon across the 3 events' phase means
    _evmeans=np.array([[r["Baseline"],r["TOR"],r["Action"]] for r in _rp2])
    _se=_evmeans.std(0,ddof=1)/np.sqrt(_evmeans.shape[0])
    ax.fill_between(xpos, np.array(mavg)-_se, np.array(mavg)+_se,
                    color="0.5", alpha=0.18, zorder=2, label="Average +/-SE")
    ax.plot(xpos,mavg,linestyle="--",color="black",lw=2.6,zorder=6,
            marker="D",ms=9,markerfacecolor="black",markeredgecolor="white",markeredgewidth=1.1,
            label="Average")
    _,pbt=_perm_two_region(edavg,Ra["TOR"],Ra["Baseline"])
    _,pta=_perm_two_region(edavg,Ra["Action"],Ra["TOR"])'''
assert old_avg in s, "average plot block not found"
s=s.replace(old_avg,new_avg,1)

ast.parse(s)
nb["cells"][i]["source"]=s
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {i}: Okabe-Ito events; average = black dashed + diamonds + SE ribbon.")
