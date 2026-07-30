import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA_temp_16.6.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "PHASE-MEAN PLOT" in "".join(c["source"]): break
s="".join(nb["cells"][i]["source"])

# 1) xlab: mark TOR/Action windows as (avg)
old_xlab = 'xlab=[f"Baseline\\n[-4, 0)", f"TOR\\n[0, {_rt_mean:.1f})", f"Action\\n[{_rt_mean:.1f}, 4]"]'
new_xlab = 'xlab=["Baseline\\n[-4, 0)", f"TOR\\n[0, {_rt_mean:.1f}) avg", f"Action\\n[{_rt_mean:.1f}, 4] avg"]'
assert old_xlab in s, "xlab line not found"
s=s.replace(old_xlab,new_xlab,1)

# 2) title: add per-event TOR boundary note (each event uses its own RT)
old_title = 'ax.set_title("ED across phases: Baseline / Take-over request / Action")'
new_title = ('_rtmap={_EENAME[e]:_gold_ED_store[e]["rt"] for e in _evk}\n'
             '_rtnote=", ".join(f"{nm.split()[0]} {_rtmap.get(nm, float(\'nan\')):.1f}s" for nm in [r["event"] for r in _rp2])\n'
             'ax.set_title("ED across phases: Baseline / Take-over request / Action\\n"\n'
             '             f"(x at across-event avg; per-event TOR boundary: {_rtnote})", fontsize=10.5)')
assert old_title in s, "title line not found"
s=s.replace(old_title,new_title,1)

ast.parse(s)
nb["cells"][i]["source"]=s
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {i}: xlab marked (avg); title lists per-event TOR boundaries.")
