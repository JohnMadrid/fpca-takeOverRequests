import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

# ---- Group cell (122): default to lossless ----
s = "".join(nb["cells"][122]["source"]) if isinstance(nb["cells"][122]["source"], list) else nb["cells"][122]["source"]

# 1) change default PVE to lossless (1.0 keeps everything)
s = s.replace("_GR_PVE = 0.95",
              "_GR_PVE = 1.0     # LOSSLESS default (keep all components). "
              "Set <1.0 (e.g. 0.95) to truncate.")

# 2) make the truncation robust at pve=1.0: keep all non-zero components.
old_trunc = '''        tot = evals.sum()
        if tot <= 0:
            stacked.append(Xc[:, :1]*0); continue
        pve_cum = np.cumsum(evals)/tot
        nk = int(np.searchsorted(pve_cum, pve) + 1)
        nk = max(1, min(nk, len(evals)))'''
new_trunc = '''        tot = evals.sum()
        if tot <= 0:
            stacked.append(Xc[:, :1]*0); continue
        if pve >= 1.0:
            nk = int(np.sum(evals > 1e-12))     # lossless: all non-zero comps
        else:
            pve_cum = np.cumsum(evals)/tot
            nk = int(np.searchsorted(pve_cum, pve) + 1)
        nk = max(1, min(nk, len(evals)))'''
assert old_trunc in s, "group truncation block not found"
s = s.replace(old_trunc, new_trunc)
ast.parse(s)
nb["cells"][122]["source"] = s
print("Group cell: default lossless (PVE=1.0), all non-zero components kept.")

# ---- A1 scree cell (119): already plots elbow + 95% markers; nothing to change
# it does not truncate, only shows scree. Confirm markers exist.
sa = "".join(nb["cells"][119]["source"]) if isinstance(nb["cells"][119]["source"], list) else nb["cells"][119]["source"]
has_elbow = "elbow @" in sa
has_95 = "95% @" in sa
print(f"A1 scree: elbow marker={has_elbow}, 95% marker={has_95} (kept as reference).")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Saved.")
