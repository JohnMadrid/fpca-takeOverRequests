import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

i=None
for k,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "ED PRE vs POST -- GOLD STANDARD" in "".join(c["source"]): i=k; break
assert i is not None
s="".join(nb["cells"][i]["source"])

# 1) init store + record per event
s=s.replace("_e_curves=[]; _e_secs=None","_e_curves=[]; _e_secs=None; _gold_ED_store={}")
old='''    ed=_ed_curve_gold(arr); _e_curves.append(ed); _e_secs=secs
    pre=secs<0; post=secs>=0; pm=ed[pre].mean()
    dur=(post&(ed<pm)).sum()*(secs[1]-secs[0])'''
new='''    ed=_ed_curve_gold(arr); _e_curves.append(ed); _e_secs=secs
    _gold_ED_store[_e]=dict(secs=secs,ed=ed,rt=_mean_rt_evt(_e))
    pre=secs<0; post=secs>=0; pm=ed[pre].mean()
    dur=(post&(ed<pm)).sum()*(secs[1]-secs[0])'''
assert old in s
s=s.replace(old,new,1)

means_block = r'''

# ============================================================
# REGION MEANS -- two contrasts (used by the perm test in the next cell)
#  Contrast 1 (pre vs post):  Pre [-4,0)  vs  Post [0,+4]
#  Contrast 2 (3 phases):     Baseline [-4,0) | TOR [0,meanRT) | Action [meanRT,+4]
#  TOR/Action boundary = fixed mean RT_Steering per event (option A): ED(t) is a
#  cross-driver quantity, so the boundary must be common across drivers.
#  Region value = mean ED over region; also report per-region level + min/max.
# ============================================================
def _regions(secs, rt):
    return dict(
        Pre =(secs>=-4)&(secs<0),
        Post=(secs>=0)&(secs<=4),
        Baseline=(secs>=-4)&(secs<0),
        TOR =(secs>=0)&(secs<rt),
        Action=(secs>=rt)&(secs<=4),
    )

print("\n"+"="*72)
print("REGION MEANS (gold ED). Contrast2 boundary = mean RT_Steering per event.")
print("="*72)
print(f"{'event':<14}{'region':<10}{'meanED':>8}{'minED':>8}{'maxED':>8}{'window(s)':>16}")
print("-"*64)
_gold_region_means={}
for _e in _EEVENTS:
    if _e not in _gold_ED_store: continue
    d=_gold_ED_store[_e]; secs=d['secs']; ed=d['ed']; rt=d['rt']
    R=_regions(secs,rt); _gold_region_means[_e]={}
    for name in ['Pre','Post','Baseline','TOR','Action']:
        m=R[name]
        if m.sum()==0:
            print(f"{_EENAME[_e]:<14}{name:<10}{'(empty)':>8}"); continue
        seg=ed[m]; w=secs[m]
        _gold_region_means[_e][name]=float(seg.mean())
        print(f"{_EENAME[_e]:<14}{name:<10}{seg.mean():>8.3f}{seg.min():>8.3f}{seg.max():>8.3f}"
              f"{f'[{w.min():.2f},{w.max():.2f}]':>16}")
    print(f"{'':14}{'(meanRT)':<10}{rt:>8.2f}")
    print("-"*64)
print("Contrast1 stat = Post-Pre. Contrast2: pairwise phase diffs + omnibus (next cell).")
'''
ast.parse(s+means_block)
nb["cells"][i]["source"]=s+means_block
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {i}: region-means block added + _gold_ED_store/_gold_region_means.")
