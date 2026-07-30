import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

# --- A1: add per-driver temporal demeaning (operation C) to _mf_twostep_raw ---
a1=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "def _mf_twostep_raw" in "".join(c["source"]):
        a1=i; break
assert a1 is not None, "A1 _mf_twostep_raw not found"
s="".join(nb["cells"][a1]["source"])

old='''    arr3d = _mf_selfstd(arr3d.astype(float))
    W = arr3d.shape[0]'''
new='''    # OPERATION C (gold standard): per-driver per-channel temporal demeaning,
    # to match snapshot PCA (StandardScaler per driver). Removes static per-driver
    # offsets (VR seating posture / eye-tracker calibration) so ED reflects
    # trajectory SHAPE, not constant offset. Applied BEFORE pooled self-std.
    arr3d = arr3d.astype(float)
    arr3d = arr3d - arr3d.mean(axis=1, keepdims=True)
    arr3d = _mf_selfstd(arr3d)
    W = arr3d.shape[0]'''
assert old in s, "twostep body anchor not found"
s=s.replace(old,new,1)
ast.parse(s)
nb["cells"][a1]["source"]=s
print(f"A1 (cell {a1}): operation C added to _mf_twostep_raw.")

# --- COORDINATION TRIAL: _co_blocks builds its own stacked scores (does NOT call
# _mf_twostep_raw), so add C there too for consistency. ---
tr=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "def _co_blocks" in "".join(c["source"]):
        tr=i; break
if tr is not None:
    s2="".join(nb["cells"][tr]["source"])
    # _co_blocks begins by self-std then per-channel loop; insert C before its selfstd.
    # find its function and its first standardization call
    if "_co_selfstd" in s2:
        oldb="    arr3d = _co_selfstd(arr3d)"
        newb=("    # OPERATION C: per-driver temporal demeaning (match _mf_twostep_raw)\n"
              "    arr3d = arr3d.astype(float)\n"
              "    arr3d = arr3d - arr3d.mean(axis=1, keepdims=True)\n"
              "    arr3d = _co_selfstd(arr3d)")
        cnt=s2.count(oldb)
        assert cnt>=1, "_co_selfstd call not found in trial cell"
        s2=s2.replace(oldb,newb)   # apply to all occurrences (blocks + offdiag)
        ast.parse(s2)
        nb["cells"][tr]["source"]=s2
        print(f"Trial (cell {tr}): operation C added to {cnt} self-std site(s).")
    else:
        print(f"Trial (cell {tr}): no _co_selfstd found; SKIPPED (check manually).")
else:
    print("Trial cell not found; skipped.")

# --- OPTION 3 cell: _selfstd then _twostep_blocks -> add C before its _selfstd ---
op=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "_ed_plain_and_equalized" in "".join(c["source"]):
        op=i; break
if op is not None:
    s3="".join(nb["cells"][op]["source"])
    olde="    a=_selfstd(arr3d)"
    newe=("    a=arr3d.astype(float)\n"
          "    a=a - a.mean(axis=1, keepdims=True)   # OPERATION C: per-driver demean\n"
          "    a=_selfstd(a)")
    if olde in s3:
        s3=s3.replace(olde,newe,1); ast.parse(s3); nb["cells"][op]["source"]=s3
        print(f"Option 3 (cell {op}): operation C added.")
    else:
        print(f"Option 3 (cell {op}): anchor not found; SKIPPED.")
else:
    print("Option 3 cell not found; skipped.")

json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved. Total cells: {len(nb['cells'])}.")
