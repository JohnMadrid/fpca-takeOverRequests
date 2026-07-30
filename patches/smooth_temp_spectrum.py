import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA_temp_16.6.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "EIGENVALUE SPECTRUM OVER TIME (GOLD" in "".join(c["source"]): break
s="".join(nb["cells"][i]["source"])

# add a display smoothing helper after the imports
anchor='_N_PERM=5000; _PERM_SEED=0'
helper=anchor+'''
from scipy.ndimage import gaussian_filter1d as _g1d
def _sp_disp(y, sigma=1.0):
    """Display-only smoothing (sigma=1). Stats/heatmaps use raw values."""
    return _g1d(np.asarray(y,float), sigma=sigma)'''
assert anchor in s
s=s.replace(anchor,helper,1)

# per-event spectrum lines -> smoothed for display
old1='for k in range(_SP_M): ax0.plot(secs,r[k],color=_PC_COLORS[k],lw=2,alpha=0.9,label=f"PC{k+1}")'
new1='for k in range(_SP_M): ax0.plot(secs,_sp_disp(r[k]),color=_PC_COLORS[k],lw=2,alpha=0.9,label=f"PC{k+1}")'
assert old1 in s
s=s.replace(old1,new1,1)

# averaged spectrum line + ribbon -> smoothed for display
old2='ax0.plot(ref,m,color=_PC_COLORS[k],lw=2,alpha=0.9,label=f"PC{k+1}"); ax0.fill_between(ref,m-se,m+se,color=_PC_COLORS[k],alpha=0.18,lw=0)'
new2='ax0.plot(ref,_sp_disp(m),color=_PC_COLORS[k],lw=2,alpha=0.9,label=f"PC{k+1}"); ax0.fill_between(ref,_sp_disp(m-se),_sp_disp(m+se),color=_PC_COLORS[k],alpha=0.18,lw=0)'
assert old2 in s
s=s.replace(old2,new2,1)

ast.parse(s)
nb["cells"][i]["source"]=s
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {i}: display-only sigma=1 smoothing on spectrum lines (per-event + averaged). Stats/heatmaps raw.")
