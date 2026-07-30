"""PCA demo (looping). A fixed point cloud shaped as a DIAGONAL cigar (elongated
along a 3D diagonal). The camera starts looking ~down the long axis so the cloud
reads as a barely-upward ellipsoid (egg). It then rotates sideways and upward,
revealing the diagonal cigar, and the cloud morphs slowly until its long axis is
HORIZONTAL (PC1). When it reaches horizontal, the PC1/PC2/PC3 axes fade in, sized
to the actual extent of the dots (the cloud is confined to that box) and rendered
in 3D so points crossing an axis sit in front of it. Then it reverses and loops
seamlessly. Slate/clay palette, deck background.
Saves plots/ellipsoid/manifold_pca_demo.gif."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa

BG="#e9ece6"
rng = np.random.default_rng(11)
N = 300
C_A="#5b7a99"; C_B="#b58a6a"; AXC="#6f6f6f"     # slate blue / clay-rust; darker grey axes
AX_ALPHA=0.75                                    # PC axes (30% darker than before: 0.55->0.75 alpha + darker grey)

def smooth(x): return x*x*(3-2*x)

# ---------------------------------------------------------------------------
# The cloud is built ONCE in a canonical frame where the long axis is the data
# x-axis. We then ROTATE the whole cloud over the animation: start as a diagonal
# cigar, morph to having the long axis horizontal. Two equivalent ways to get
# "diagonal" vs "horizontal": rotate the cloud, or rotate the camera. We rotate
# the cloud so the long axis lands exactly horizontal on screen at the end, and
# orbit the camera slightly for the "sideways + up" reveal.
# ---------------------------------------------------------------------------
# canonical cigar: long along x, thin in y,z (PC2,PC3 spreads slightly unequal)
def make_cloud(n, center):
    t = np.clip(rng.normal(center, 0.95, n), -2.4, 2.4)
    x = t * 2.25                       # long axis (PC1)
    y = rng.normal(0, 0.93, n)         # PC2 spread (50% fatter)
    z = rng.normal(0, 0.75, n)         # PC3 spread (50% fatter)
    return np.column_stack([x, y, z])
BASE_A = make_cloud(N, +0.55)
BASE_B = make_cloud(N, -0.55)
BASE = np.vstack([BASE_A, BASE_B])

# rotation that tilts the cigar into a DIAGONAL (start) vs leaves it HORIZONTAL (end).
def rot_about(axis, ang):
    a = axis/np.linalg.norm(axis); c,s = np.cos(ang), np.sin(ang)
    x,y,z = a
    return np.array([[c+x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s, c+y*y*(1-c),   y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]])

# at start the long axis is tilted up (+z) and into depth (+y) -> diagonal cigar
START_TILT = rot_about(np.array([0.0,1.0,1.0]), np.deg2rad(38))

def cloud_at(s):
    """s in [0,1]: 0 = diagonal (tilted), 1 = horizontal (long axis = x)."""
    R = rot_about(np.array([0.0,1.0,1.0]), np.deg2rad(38)*(1-s))
    return BASE @ R.T

NA = BASE_A.shape[0]

# camera: start looking NEARLY down the long axis so the cigar foreshortens into a
# barely-upward egg (the exact look-down angle is elev=-25.8, azim=28.9; we offset
# slightly so it reads as an egg, not a perfect dot). End at near side-on view.
A_azim, A_elev = 40, -13     # reveal angle: cigar foreshortened -> egg/ellipsoid
# NOT perfectly side-on: offset azim/elev so PC3 (depth) stays visible and the
# LDA plane catches face area. PC1 still reads essentially horizontal.
B_azim, B_elev = -78, 12

# axis box: a bit LARGER than the cloud so the dots live inside the axis cube.
FINAL = cloud_at(1.0)
DX = np.abs(FINAL[:,0]).max()        # actual dot extents
DY = np.abs(FINAL[:,1]).max()
DZ = np.abs(FINAL[:,2]).max()
EX = DX*1.30                          # PC1 length (x): ~30% past the cloud
EZ = DZ*1.30                          # PC2 length (z / vertical on screen)
EY = DY*1.30                          # PC3 length (y / depth)

# LDA plane: y-z plane at x=0 (perpendicular to PC1), sized to the DOTS (a wall
# through the cloud), not to the larger axis cube.
_gy = np.linspace(-DY*1.05, DY*1.05, 2)
_gz = np.linspace(-DZ*1.05, DZ*1.05, 2)
_PY, _PZ = np.meshgrid(_gy, _gz)
_PX = np.zeros_like(_PY)
LDAC = "#8fb4d6"          # neutral light blue (was warm red)

# ---------------------------------------------------------------------------
# timeline (33 fps): A(egg) -> B(reveal+morph, axes fade in) -> 2s pause ->
# axes fade OUT -> plane fades IN -> 2s hold -> plane fades OUT -> back -> A.
# Axes and plane never coexist.
# ---------------------------------------------------------------------------
TRAVEL  = 170    # frames A->B (rotation + morph), and B->A
HOLDB   = 198    # ~6s pause at the horizontal cigar with PC axes shown (3x longer)
AXOUT   = 33     # ~1s PC axes fade out
PLANEIN = 33     # ~1s LDA plane fade in
HOLDP   = 198    # ~6s hold with the LDA plane (3x longer)
PLANEOUT= 33     # ~1s LDA plane fade out
t0 = TRAVEL
t1 = t0 + HOLDB
t2 = t1 + AXOUT
t3 = t2 + PLANEIN
t4 = t3 + HOLDP
t5 = t4 + PLANEOUT
CYCLE  = t5 + TRAVEL
FRAMES = CYCLE

def phase(frame):
    """Return (s 0..1 reveal/morph, axis_alpha 0..1, plane_alpha 0..1)."""
    if frame < t0:                                   # A -> B, axes fade in at the end
        s = smooth(frame/TRAVEL)
        return s, smooth(max(0,(frame-(TRAVEL-28))/28)), 0.0
    if frame < t1:                                   # 2s pause, axes only
        return 1.0, 1.0, 0.0
    if frame < t2:                                   # PC axes fade out
        return 1.0, smooth(1-(frame-t1)/AXOUT), 0.0
    if frame < t3:                                   # LDA plane fades in
        return 1.0, 0.0, smooth((frame-t2)/PLANEIN)
    if frame < t4:                                   # 2s hold, plane only
        return 1.0, 0.0, 1.0
    if frame < t5:                                   # LDA plane fades out
        return 1.0, 0.0, smooth(1-(frame-t4)/PLANEOUT)
    f = frame-t5                                     # B -> A
    s = smooth(1 - f/TRAVEL)
    return s, 0.0, 0.0

fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection="3d")
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

def draw(frame):
    ax.clear()
    s, axis_a, plane_a = phase(frame)
    P = cloud_at(s)
    azim = A_azim + (B_azim - A_azim)*s
    elev = A_elev + (B_elev - A_elev)*s
    # LDA plane (perpendicular to PC1) behind the points
    if plane_a > 0.01:
        ax.plot_surface(_PX,_PY,_PZ, color=LDAC, alpha=0.385*plane_a,
                        edgecolor=LDAC, linewidth=2.0, shade=False, zorder=0)
    # PC axes FIRST (low zorder) so points drawn after sit in front of them.
    if axis_a > 0.01:
        for (dx,dy,dz,lbl,ext) in [(EX,0,0,"PC1",EX),(0,0,EZ,"PC2",EZ),(0,EY,0,"PC3",EY)]:
            ax.quiver(0,0,0, dx,dy,dz, color=AXC, lw=1.3,
                      arrow_length_ratio=0.12, alpha=axis_a*AX_ALPHA, zorder=1)
            ax.text(dx*1.12, dy*1.12, dz*1.12, lbl, color=AXC, fontsize=11,
                    fontweight="bold", ha="center", va="center", alpha=axis_a*AX_ALPHA, zorder=2)
    ax.scatter(P[:NA,0],P[:NA,1],P[:NA,2], s=15, c=C_A, alpha=0.65,
               edgecolors="none", depthshade=True, zorder=5)
    ax.scatter(P[NA:,0],P[NA:,1],P[NA:,2], s=15, c=C_B, alpha=0.65,
               edgecolors="none", depthshade=True, zorder=5)
    ax.view_init(elev=elev, azim=azim)
    L=3.2
    ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_zlim(-L,L)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # fully transparent panes so only the deck-coloured figure background shows
    # (filled panes overlap at angles and create a faint darker box = the "weird bg").
    for _pn in (ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane):
        _pn.set_visible(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_alpha(0.0)
    ax.set_axis_off()
    ax.grid(False)
    return ax,

OUT = os.path.join("..", "..", "plots", "ellipsoid")
os.makedirs(OUT, exist_ok=True)

# Render every frame manually. PillowWriter/FuncAnimation was dropping the
# static hold frames (dedup), which made the held PCA/LDA states flash by. We
# grab each frame off the canvas and write all of them with an explicit per-frame
# duration so every pause frame survives.
from PIL import Image
GIF_MS = 30
frames=[]
for fr in range(FRAMES):
    draw(fr)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frames.append(Image.fromarray(buf[:,:,:3].copy()))
frames[0].save(os.path.join(OUT,"manifold_pca_demo.gif"),
               save_all=True, append_images=frames[1:],
               duration=GIF_MS, loop=0)   # no disposal=2 -> identical frames not collapsed
print("saved plots/ellipsoid/manifold_pca_demo.gif", len(frames), "frames")

# verification stills straight from draw (no GIF frame-merging)
draw(0);                    fig.savefig(os.path.join(OUT,"_pca_start.png"), facecolor=BG)
draw(TRAVEL//2);            fig.savefig(os.path.join(OUT,"_pca_mid.png"),   facecolor=BG)
draw(t0+30);                fig.savefig(os.path.join(OUT,"_pca_end.png"),   facecolor=BG)
draw(t3+HOLDP//2);          fig.savefig(os.path.join(OUT,"_pca_plane.png"),facecolor=BG)
print("saved verification stills")
