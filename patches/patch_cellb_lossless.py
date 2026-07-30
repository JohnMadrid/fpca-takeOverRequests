import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

# ---- A1 (119): add lossless branch to _mf_twostep_raw ----
sa = "".join(nb["cells"][119]["source"]) if isinstance(nb["cells"][119]["source"], list) else nb["cells"][119]["source"]
old = '''        if fixed_k is not None:
            n_keep = int(min(fixed_k, len(evals)))
        else:
            tot = evals.sum()
            pve_cum = np.cumsum(evals) / tot if tot > 0 else np.zeros_like(evals)
            n_keep = int(np.searchsorted(pve_cum, pve) + 1)'''
new = '''        if fixed_k is not None:
            n_keep = int(min(fixed_k, len(evals)))
        elif pve is not None and pve >= 1.0:
            n_keep = int(np.sum(evals > 1e-12))     # LOSSLESS: all non-zero comps
        else:
            tot = evals.sum()
            pve_cum = np.cumsum(evals) / tot if tot > 0 else np.zeros_like(evals)
            n_keep = int(np.searchsorted(pve_cum, pve) + 1)'''
assert old in sa, "A1 _mf_twostep_raw truncation block not found"
sa = sa.replace(old, new)
ast.parse(sa)
nb["cells"][119]["source"] = sa
print("A1 (119): _mf_twostep_raw now supports lossless (pve>=1.0).")

# ---- Cell B (121): default _MF_PVE = 1.0 (lossless), independent of A2 ----
sb = "".join(nb["cells"][121]["source"]) if isinstance(nb["cells"][121]["source"], list) else nb["cells"][121]["source"]
# insert an override at the top of the executable body (after the header comment)
# Find the first non-comment code line; prepend the override there.
marker = "# Uses the truncation set in CELL A2 (_MF_PVE or _MF_FIXED_K)."
override = ("# LOSSLESS by default for Cell B (overrides A2). Set <1.0 to truncate.\n"
            "_MF_PVE = 1.0\n"
            "_MF_FIXED_K = None\n")
if marker in sb:
    sb = sb.replace(marker, marker + "\n" + override, 1)
else:
    # fallback: prepend after first line
    lines = sb.split("\n", 1)
    sb = lines[0] + "\n" + override + (lines[1] if len(lines) > 1 else "")
ast.parse(sb)
nb["cells"][121]["source"] = sb
print("Cell B (121): _MF_PVE=1.0 lossless default.")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Saved.")
