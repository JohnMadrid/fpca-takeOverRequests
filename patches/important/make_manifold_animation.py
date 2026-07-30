"""Animated 3D cloud morphing from an isotropic sphere into a 1D 'cigar' aligned
along a linear combination of all 3 axes (Environmental constraints, Cognitive
goals, Biomechanical limits). Two subtle-coloured groups settle toward opposite
ends of the cigar with moderate overlap (LDA foreshadowing). Saves a GIF."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa

rng = np.random.default_rng(7)
N = 260                      # points per group
# subtle, non-neon palette
C_A = "#5b7a99"              # muted slate blue
C_B = "#b58a6a"              # muted clay/tan
AX  = "#3a3a3a"

# --- start: isotropic sphere (two interleaved groups, no separation yet) ---
def sphere(n):
    p = rng.normal(0, 1.0, (n, 3))
    return p
S_A = sphere(N); S_B = sphere(N)

# --- end: cigar from near-low (by viewer at Environmental) up-right-forward ---
# axes: x=Environmental, y=Cognitive, z=Biomechanical
# near-low -> far-up-right: +y (rightward), +z (up), slight +x (forward/away)
# near-left -> far-right diagonal: strong +y (rightward) + depth, moderate +z
u = np.array([1.0, 0.15, 1.0]); u = u / np.linalg.norm(u)  # max x-z diagonal
tmp = np.array([0.0, 0.0, 1.0])
v1 = np.cross(u, tmp); v1 /= np.linalg.norm(v1)
v2 = np.cross(u, v1);  v2 /= np.linalg.norm(v2)
def cigar(n, center_t):
    # t = position along the cigar axis; groups biased to opposite ends but overlap
    t = rng.normal(center_t, 0.95, n)
    t = np.clip(t, -2.2, 2.2)
    # 50% thinner radius than before (0.85/0.70 -> 0.42/0.35)
    r1 = rng.normal(0, 0.554, n); r2 = rng.normal(0, 0.462, n)
    bend = 0.12*(t**2 - 1.0)
    return (t[:,None]*u[None,:]*1.862 + (r1+bend)[:,None]*v1[None,:] + r2[:,None]*v2[None,:])
E_A = cigar(N, +0.65)       # group A toward + end (30% more mixed: centers 0.9->0.65)
E_B = cigar(N, -0.65)        # group B toward - end

# --- morph schedule: hold sphere, morph, hold cigar; plus slow rotation ---
def ease(x):                 # smoothstep
    return x*x*(3-2*x)

# timeline (33 fps): hold -> labels fade in (LEAD before collapse) -> morph -> 3s end hold
HOLD0  = 30          # initial sphere with first labels
LEAD   = 25          # fade-in starts; collapse begins slightly AFTER this
FADE   = 90          # constraint-label fade-in duration
MORPH  = 260         # collapse into the cigar
HOLD1  = 100         # ~3s hold at the final cigar state
FRAMES = HOLD0 + LEAD + MORPH + HOLD1

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
fig.patch.set_facecolor("#e9ece6"); ax.set_facecolor("#e9ece6")

def morph_alpha(frame):
    m0 = HOLD0 + LEAD                  # collapse begins slightly after fade-in starts
    if frame < m0: return 0.0
    if frame < m0 + MORPH: return ease((frame - m0)/MORPH)
    return 1.0

def label2_alpha(frame):
    # second labels fade in softly, starting before the collapse
    if frame < HOLD0: return 0.0
    if frame < HOLD0 + FADE: return ease((frame - HOLD0)/FADE)
    return 1.0

# label positions (figure coords). First label on top, second fades below/next to it.
# "Cognitive strategies" nudged to sit right under its axis corner.
_LBL = [  # (x, y, primary, secondary, rotation, secondary_dy)
    (0.335, 0.165, "Afforded interactions", "Environmental constraints", 0, -0.040),  # diagonal w/ axis
    (0.730, 0.225, "Cognitive strategies",  "Task rules",                0,  -0.040),  # diagonal w/ axis
    (0.885, 0.560, "Movements",            "Biomechanical limits",      0,  -0.036),     # 2nd to the RIGHT
]
_txt = []  # persistent text handles, updated per frame

def _ensure_text():
    if _txt: return
    for (x,y,p,sec,rot,dy) in _LBL:
        t1 = fig.text(x, y, p, fontsize=9.5, color=AX, ha="center", va="center", rotation=rot, fontweight="bold")
        if rot == 90:  # Movements: secondary sits to the RIGHT, same rotation
            t2 = fig.text(x+0.055, y, sec, fontsize=10, color="#5a5a5a", ha="center", va="center", rotation=rot)
        elif rot == -25:  # Cognitive strategies: secondary parallel, offset below the diagonal
            t2 = fig.text(x, y+dy, sec, fontsize=10, color="#5a5a5a", ha="center", va="center", rotation=rot)
        else:          # secondary below
            t2 = fig.text(x, y+dy, sec, fontsize=10, color="#5a5a5a", ha="center", va="center", rotation=rot)
        _txt.append((t1,t2))

def draw(frame):
    ax.clear()
    _ensure_text()
    a = morph_alpha(frame)
    PA = (1-a)*S_A + a*E_A
    PB = (1-a)*S_B + a*E_B
    ax.scatter(PA[:,0],PA[:,1],PA[:,2], s=14, c=C_A, alpha=0.55, edgecolors="none", depthshade=True)
    ax.scatter(PB[:,0],PB[:,1],PB[:,2], s=14, c=C_B, alpha=0.55, edgecolors="none", depthshade=True)
    # second labels fade in softly
    a2 = label2_alpha(frame)
    for (t1,t2) in _txt:
        t1.set_alpha(1.0); t2.set_alpha(a2)
    # gentle camera orbit
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

anim = FuncAnimation(fig, draw, frames=FRAMES, interval=26, blit=False)
anim.save("manifold_morph.gif", writer=PillowWriter(fps=38), savefig_kwargs={"facecolor":"#e9ece6"})
print("saved manifold_morph.gif")
