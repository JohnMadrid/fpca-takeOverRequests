"""Expansion animation (counterpart to the collapse GIF): points start as a 1D
horizontal line and EXPLODE outward as each dimension is added, with a label
appearing per added axis (Afforded interactions -> Cognitive strategies ->
Movements). Ends as a full 3D scattered blob = dimensional explosion.
Saves manifold_expand.gif."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa

rng = np.random.default_rng(7)
N = 260
C_A = "#5b7a99"; C_B = "#b58a6a"; AX = "#3a3a3a"

# target full-3D scatter (the "exploded" high-dim state)
FULL_A = rng.normal(0, 1.0, (N,3))
FULL_B = rng.normal(0, 1.0, (N,3))
# 1D line start: keep x spread, collapse y,z to ~0 (a horizontal line along x)
def to_line(P):
    L = P.copy(); L[:,1] = 0.0; L[:,2] = 0.0
    # widen along x so it reads as a line, not a dot
    L[:,0] = P[:,0]*1.8
    return L
LINE_A = to_line(FULL_A); LINE_B = to_line(FULL_B)
# 0D dot: all points collapsed to a single location (tiny scatter so they're visible)
DOT_A = rng.normal(0, 0.02, (N,3)); DOT_B = rng.normal(0, 0.02, (N,3))
# 2D plane state: x + y spread, z still flat
PLANE_A = FULL_A.copy(); PLANE_A[:,2]=0.0; PLANE_A[:,0]*=1.4
PLANE_B = FULL_B.copy(); PLANE_B[:,2]=0.0; PLANE_B[:,0]*=1.4

def ease(x): return x*x*(3-2*x)

# timeline (33 fps): dot hold -> ->line -> +dim2 -> +dim3 -> hold blob
HOLD0 = 25          # 0D dot
STEP  = 65          # each dimension-add expansion
GAP   = 16          # brief settle between adds
HOLD1 = 90          # ~3s hold at the exploded blob
FRAMES = HOLD0 + STEP + GAP + STEP + GAP + STEP + GAP + STEP + HOLD1

# expansion phase boundaries
e0 = HOLD0                     # dot -> line (dim 1)
e1 = e0 + STEP + GAP           # line -> plane (dim 2)
e2 = e1 + STEP + GAP           # plane -> full (dim 3)
e3 = e2 + STEP + GAP           # full + symbolic "more dims" jitter

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
fig.patch.set_facecolor("#e9ece6"); ax.set_facecolor("#e9ece6")

# labels appear one per added dimension
_LBL = [
    (0.335, 0.165, "Afforded interactions", e0),   # appears as dim 1 (line) forms
    (0.730, 0.225, "Cognitive strategies",  e1),   # dim 2 (plane)
    (0.885, 0.560, "Movements",             e2),   # dim 3 (full)
]
_txt = []
def _ensure_text():
    if _txt:
        return
    for (x,y,p,f0) in _LBL:
        _txt.append((fig.text(x,y,p,fontsize=10,color=AX,ha="center",va="center",fontweight="bold"), f0))

def lerp(a,b,t): return (1-t)*a + t*b

def state(frame):
    # DOT -> LINE (dim1) -> PLANE (dim2) -> FULL (dim3) -> jitter
    if frame < e0:
        return DOT_A, DOT_B
    if frame < e1:                       # dot -> line (add dim 1)
        t = ease(min(1,(frame-e0)/STEP))
        return lerp(DOT_A,LINE_A,t), lerp(DOT_B,LINE_B,t)
    if frame < e2:                       # line -> plane (add dim 2)
        t = ease(min(1,(frame-e1)/STEP))
        return lerp(LINE_A,PLANE_A,t), lerp(LINE_B,PLANE_B,t)
    if frame < e3:                       # plane -> full (add dim 3)
        t = ease(min(1,(frame-e2)/STEP))
        return lerp(PLANE_A,FULL_A,t), lerp(PLANE_B,FULL_B,t)
    return FULL_A, FULL_B                 # hold steady full blob (no jitter)

def draw(frame):
    ax.clear(); _ensure_text()
    PA,PB = state(frame)
    ax.scatter(PA[:,0],PA[:,1],PA[:,2], s=14, c=C_A, alpha=0.55, edgecolors="none", depthshade=True)
    ax.scatter(PB[:,0],PB[:,1],PB[:,2], s=14, c=C_B, alpha=0.55, edgecolors="none", depthshade=True)
    # labels fade in at their scheduled frame
    for (t,f0) in _txt:
        a = ease(min(1, max(0,(frame-f0)/22)))
        t.set_alpha(a)
    azim = -58 + 6.5*np.sin(2*np.pi*frame/FRAMES)
    elev = 18 + 1.95*np.sin(2*np.pi*frame/FRAMES + 0.8)
    ax.view_init(elev=elev, azim=azim)
    L=3.2
    ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_zlim(-L,L)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for _pn in (ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane): _pn.set_facecolor("#e9ece6"); _pn.set_edgecolor("none"); _pn.set_alpha(1.0)
    ax.grid(False)
    return ax,

anim = FuncAnimation(fig, draw, frames=FRAMES, interval=30, blit=False)
anim.save("manifold_expand.gif", writer=PillowWriter(fps=33), savefig_kwargs={"facecolor":"#e9ece6"})
print("saved manifold_expand.gif")
