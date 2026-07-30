import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code" and "LDA SHADED ACCURACY PLOTS" in "".join(c["source"]): break
s="".join(nb["cells"][i]["source"])

# --- replace _grad_fill with a saturation-encoded fill ---
import re
old = re.search(r'def _grad_fill.*?(?=\ndef _shade_acc)', s, re.S).group(0)
new = '''def _grad_fill(ax, ts, acc, ch, hue_rgb=(0.85,0.20,0.30)):
    """Fill the band between chance and the accuracy curve, encoding accuracy as
    SATURATION (not brightness): near chance = desaturated/pale, high accuracy =
    fully saturated single hue, at roughly constant lightness. Colour is constant
    within a column (set by that column's accuracy), so taller/more-accurate
    columns read as more vivid."""
    import numpy as _np
    import matplotlib.colors as _mc
    acc=_np.asarray(acc); ts=_np.asarray(ts)
    lo=min(ch, float(_np.nanmin(acc))); hi=max(float(_np.nanmax(acc)), ch)
    ny=200; ygrid=_np.linspace(lo,hi,ny)
    # saturation level per column from accuracy above chance, normalised to peak
    span=max(hi-ch,1e-6)
    sat=_np.clip((acc-ch)/span,0,1)            # 0 at chance .. 1 at peak
    # base hue in HSV; keep V (lightness) high+constant, vary S by sat
    h,_,v = _mc.rgb_to_hsv(_np.array(hue_rgb))[0], None, _mc.rgb_to_hsv(_np.array(hue_rgb))[2]
    rgba=_np.zeros((ny,len(ts),4))
    for j in range(len(ts)):
        a=acc[j]
        if _np.isnan(a): continue
        col=_mc.hsv_to_rgb((h, 0.15+0.85*sat[j], 0.95))   # constant value, S scales
        y0,y1=min(ch,a),max(ch,a)
        band=(ygrid>=y0)&(ygrid<=y1)
        rgba[band,j,:3]=col
        rgba[band,j,3]=0.35+0.6*sat[j]                    # also fade alpha a touch
    ax.imshow(rgba, aspect="auto", origin="lower",
              extent=[ts[0],ts[-1],lo,hi], zorder=1, interpolation="bilinear")'''
s=s.replace(old,new,1)

# --- accuracy line: revert to a clean dark line (no white halo) ---
s=s.replace('''    ax.plot(ts,acc,color="white",lw=2.4,zorder=4)
    ax.plot(ts,acc,color="#222",lw=1.0,zorder=5)''',
            '    ax.plot(ts,acc,color="#222",lw=1.8,zorder=4)')

ast.parse(s)
nb["cells"][i]["source"]=s
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"cell {i}: saturation-encoded fill (constant lightness), dark accuracy line.")
