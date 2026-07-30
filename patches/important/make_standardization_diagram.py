"""Standalone: render a two-pipeline standardization diagram for the slide.
Saves standardization_diagram.png (and .pdf) in the project root."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13); ax.set_ylim(0, 7.6); ax.axis("off")

C_GOLD  = "#2A7F62"; C_GOLD_L  = "#d8ece4"
C_OLD   = "#B5651D"; C_OLD_L   = "#f1e3d3"
C_SHARED= "#34495E"; C_SHARED_L= "#e4e8ec"
C_RED   = "#C0392B"

def box(x,y,w,h,text,fc,ec,fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc,ec=ec,lw=1.6,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,
            fontweight="bold" if bold else "normal",zorder=3,color="#1a1a1a")

def arrow(x0,y0,x1,y1,color="#555"):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=14,
                                 lw=1.6,color=color,zorder=1))

# ---- title ----
ax.text(6.5,7.25,"Two standardization recipes for effective dimensionality (ED)",
        ha="center",fontsize=14,fontweight="bold")

# ---- shared input ----
box(5.1,6.2,2.8,0.62,"Raw car-frame curves\n(W drivers x T x 5 channels)",C_SHARED_L,C_SHARED,10,True)

# ===== LEFT: per-driver z (obsolete) =====
lx=1.4; lw=3.4
ax.text(lx+lw/2,5.85,"Per-driver z  (obsolete)",ha="center",fontsize=12,fontweight="bold",color=C_OLD)
box(lx,5.0,lw,0.6,"Per-driver CENTER + SCALE\n(StandardScaler, each driver -> unit var)",C_OLD_L,C_OLD,9)
box(lx,4.1,lw,0.6,"Per-timepoint cross-driver center\n+ per-channel z",C_OLD_L,C_OLD,9)
box(lx,3.2,lw,0.55,"5x5 covariance -> eigenvalues",C_OLD_L,C_OLD,9)
box(lx,2.4,lw,0.55,"ED = (Σλ)² / Σλ²",C_OLD_L,C_OLD,10,True)
arrow(6.5,6.2,lx+lw/2,5.62); arrow(lx+lw/2,5.0,lx+lw/2,4.72)
arrow(lx+lw/2,4.1,lx+lw/2,3.77); arrow(lx+lw/2,3.2,lx+lw/2,2.97)
# consequence
box(lx,1.35,lw,0.78,"Between-driver AMPLITUDE destroyed\nED = shape only;  inflated (~4.3)",  "#fbe9e7",C_RED,9,True)
arrow(lx+lw/2,2.4,lx+lw/2,2.15)

# ===== RIGHT: gold =====
rx=8.2; rw=3.4
ax.text(rx+rw/2,5.85,"Gold  (C + pooled)",ha="center",fontsize=12,fontweight="bold",color=C_GOLD)
box(rx,5.0,rw,0.6,"Operation C: per-driver TEMPORAL demean\n(removes posture / calibration offset)",C_GOLD_L,C_GOLD,9)
box(rx,4.1,rw,0.6,"Pooled per-channel scaling\n(common scale, amplitude kept)",C_GOLD_L,C_GOLD,9)
box(rx,3.2,rw,0.55,"per-timepoint center -> 5x5 cov -> eig",C_GOLD_L,C_GOLD,9)
box(rx,2.4,rw,0.55,"ED = (Σλ)² / Σλ²",C_GOLD_L,C_GOLD,10,True)
arrow(6.5,6.2,rx+rw/2,5.62); arrow(rx+rw/2,5.0,rx+rw/2,4.72)
arrow(rx+rw/2,4.1,rx+rw/2,3.77); arrow(rx+rw/2,3.2,rx+rw/2,2.97)
box(rx,1.35,rw,0.78,"Amplitude PRESERVED; offsets removed\nED = behaviour as executed (~3.0)","#e3f2ec",C_GOLD,9,True)
arrow(rx+rw/2,2.4,rx+rw/2,2.15)

# ---- bottom punchline ----
ax.text(6.5,0.62,"Per-driver z inflates ED and manufactures significance "
        "(e.g. Motorcyclist PC1: p=.001 vs gold p=.75).",
        ha="center",fontsize=10.5,color=C_RED,fontweight="bold")
ax.text(6.5,0.25,"Gold removes only the calibration nuisance, keeps real amplitude, "
        "and matches the MFPCA preprocessing.",
        ha="center",fontsize=10.5,color=C_GOLD)

plt.tight_layout()
fig.savefig("standardization_diagram.png",dpi=200,bbox_inches="tight")
fig.savefig("standardization_diagram.pdf",bbox_inches="tight")
print("saved standardization_diagram.png + .pdf")
