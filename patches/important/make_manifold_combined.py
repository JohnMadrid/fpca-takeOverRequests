"""Combined animation: EXPAND then COLLAPSE in one GIF.
dot(0D) -> line(1D) -> plane(2D) -> full 3D blob (dimensions add, labels appear),
hold, then the blob COLLAPSES onto a 1D diagonal cigar (constraints win).
Saves manifold_combined.gif."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa

rng = np.random.default_rng(7)
N = 260
C_A = "#5b7a99"; C_B = "#b58a6a"; AX = "#3a3a3a"

# ---- expand states ----
FULL_A = rng.normal(0,1.0,(N,3)); FULL_B = rng.normal(0,1.0,(N,3))
DOT_A  = rng.normal(0,0.02,(N,3)); DOT_B = rng.normal(0,0.02,(N,3))
def to_line(P):
    L=P.copy(); L[:,1]=0; L[:,2]=0; L[:,0]=P[:,0]*1.8; return L
LINE_A=to_line(FULL_A); LINE_B=to_line(FULL_B)
PLANE_A=FULL_A.copy(); PLANE_A[:,2]=0; PLANE_A[:,0]*=1.4
PLANE_B=FULL_B.copy(); PLANE_B[:,2]=0; PLANE_B[:,0]*=1.4

# ---- collapse target: 1D diagonal cigar (max x-z diagonal, groups to opposite ends) ----
u=np.array([1.0,0.15,1.0]); u=u/np.linalg.norm(u)
v1=np.cross(u,[0,0,1.0]); v1/=np.linalg.norm(v1)
v2=np.cross(u,v1); v2/=np.linalg.norm(v2)
def cigar(n,center_t):
    t=np.clip(rng.normal(center_t,0.95,n),-2.2,2.2)
    r1=rng.normal(0,0.554,n); r2=rng.normal(0,0.462,n); bend=0.12*(t**2-1.0)
    return t[:,None]*u[None,:]*1.862 + (r1+bend)[:,None]*v1[None,:] + r2[:,None]*v2[None,:]
CIG_A=cigar(N,+0.65); CIG_B=cigar(N,-0.65)

def ease(x): return x*x*(3-2*x)
def lerp(a,b,t): return (1-t)*a+t*b

# ---- timeline ----
HOLD0=30; STEP=55; GAP=45          # expand: dot->line->plane->full (1s extra settle after each axis)
HOLDF=83                            # hold full blob (50% longer pause before collapse)
COLL=200                            # collapse blob->cigar
HOLD1=340                           # final hold (kept so total stays under 1000 frames)
FADE=45                             # fade the cigar out so the loop restarts cleanly
e0=HOLD0; e1=e0+STEP+GAP; e2=e1+STEP+GAP; e3=e2+STEP+GAP   # expand boundaries
cstart=e3+HOLDF                     # collapse begins after holding the blob
hend=cstart+COLL+HOLD1             # end of the final cigar hold (fade starts here)
FRAMES=hend+FADE

fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection="3d")
fig.patch.set_facecolor("#e9ece6"); ax.set_facecolor("#e9ece6")
# primary labels appear as dims add (expand); secondary constraint labels fade in
# just before the collapse begins.
_LBL=[(0.335,0.165,"Afforded interactions",e0,"Environmental constraints",-0.040),
      (0.730,0.225,"Cognitive strategies", e1,"Task rules",               -0.040),
      (0.885,0.560,"Movements",            e2,"Biomechanical limits",     -0.036)]
LEAD2=20                              # secondary labels start fading this many frames before collapse
_txt=[]; _txt2=[]
def _ensure_text():
    if _txt: return
    for (x,y,p,f0,sec,dy) in _LBL:
        _txt.append((fig.text(x,y,p,fontsize=10,color=AX,ha="center",va="center",fontweight="bold"),f0))
        _txt2.append(fig.text(x,y+dy,sec,fontsize=10,color="#5a5a5a",ha="center",va="center"))

def state(frame):
    if frame < e0:  return DOT_A,DOT_B
    if frame < e1:  t=ease(min(1,(frame-e0)/STEP)); return lerp(DOT_A,LINE_A,t),lerp(DOT_B,LINE_B,t)
    if frame < e2:  t=ease(min(1,(frame-e1)/STEP)); return lerp(LINE_A,PLANE_A,t),lerp(LINE_B,PLANE_B,t)
    if frame < e3:  t=ease(min(1,(frame-e2)/STEP)); return lerp(PLANE_A,FULL_A,t),lerp(PLANE_B,FULL_B,t)
    if frame < cstart: return FULL_A,FULL_B                      # hold blob
    if frame < cstart+COLL:                                      # collapse blob -> cigar
        t=ease(min(1,(frame-cstart)/COLL)); return lerp(FULL_A,CIG_A,t),lerp(FULL_B,CIG_B,t)
    return CIG_A,CIG_B                                           # hold cigar

def draw(frame):
    ax.clear(); _ensure_text()
    PA,PB=state(frame)
    # global fade-out at the very end so the cigar+labels dissolve before the loop restarts
    g = 1.0 if frame < hend else max(0.0, 1.0-(frame-hend)/FADE)
    ax.scatter(PA[:,0],PA[:,1],PA[:,2],s=14,c=C_A,alpha=0.55*g,edgecolors="none",depthshade=True)
    ax.scatter(PB[:,0],PB[:,1],PB[:,2],s=14,c=C_B,alpha=0.55*g,edgecolors="none",depthshade=True)
    for (t,f0) in _txt: t.set_alpha(ease(min(1,max(0,(frame-f0)/22)))*g)
    # secondary constraint labels fade in just before / during the collapse
    a2 = ease(min(1, max(0,(frame-(cstart-LEAD2))/60)))
    for t2 in _txt2: t2.set_alpha(a2*g)
    ax.view_init(elev=18+1.95*np.sin(2*np.pi*frame/FRAMES+0.8), azim=-58+6.5*np.sin(2*np.pi*frame/FRAMES))
    L=3.2; ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_zlim(-L,L)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for _pn in (ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane): _pn.set_facecolor("#e9ece6"); _pn.set_edgecolor("none"); _pn.set_alpha(1.0)
    ax.grid(False); return ax,

# Render every frame manually and save with summed durations, so the long final
# hold actually plays (PillowWriter drops/dedups static frames -> flashes by).
import os
from PIL import Image
OUT=os.path.join("..","..","plots","ellipsoid"); os.makedirs(OUT,exist_ok=True)
frames=[]
for fr in range(FRAMES):
    draw(fr); fig.canvas.draw()
    buf=np.asarray(fig.canvas.buffer_rgba())
    frames.append(Image.fromarray(buf[:,:,:3].copy()))
frames[0].save(os.path.join(OUT,"manifold_combined.gif"),
               save_all=True,append_images=frames[1:],duration=30,loop=0)
print("saved plots/ellipsoid/manifold_combined.gif",len(frames),"frames")
