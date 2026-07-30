import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA_temp_16.6.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "PHASE-MEAN PLOT" in "".join(c["source"]): break
s="".join(nb["cells"][i]["source"]) if isinstance(nb["cells"][i]["source"],list) else nb["cells"][i]["source"]

# revert xlab (remove " avg")
cur_xlab = 'xlab=["Baseline\\n[-4, 0)", f"TOR\\n[0, {_rt_mean:.1f}) avg", f"Action\\n[{_rt_mean:.1f}, 4] avg"]'
orig_xlab = 'xlab=[f"Baseline\\n[-4, 0)", f"TOR\\n[0, {_rt_mean:.1f})", f"Action\\n[{_rt_mean:.1f}, 4]"]'
assert cur_xlab in s, "current xlab not found"
s=s.replace(cur_xlab, orig_xlab, 1)

# revert title block (3 lines) back to single simple title
cur_title = ('_rtmap={_EENAME[e]:_gold_ED_store[e]["rt"] for e in _evk}\n'
             '_rtnote=", ".join(f"{nm.split()[0]} {_rtmap.get(nm, float(\'nan\')):.1f}s" for nm in [r["event"] for r in _rp2])\n'
             'ax.set_title("ED across phases: Baseline / Take-over request / Action\\n"\n'
             '             f"(x at across-event avg; per-event TOR boundary: {_rtnote})", fontsize=10.5)')
orig_title = 'ax.set_title("ED across phases: Baseline / Take-over request / Action")'
assert cur_title in s, "current title block not found"
s=s.replace(cur_title, orig_title, 1)

ast.parse(s)
nb["cells"][i]["source"]=s
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {i}: reverted. avg-marker present: {'avg' in s.split('xlab=')[1].split(chr(93))[0]} | rtnote present: {'_rtnote' in s}")
