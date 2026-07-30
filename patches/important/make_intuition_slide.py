"""'Intuitive prediction' bridge-slide figure: two arrows crossing exactly at
center, one rising (more demand -> more complex?), one falling (constraints
compress), with a passive->active axis underneath. Transparent for the slide."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

GREEN="#2f5233"; GREY="#6b6f6b"; ORANGE="#E8A33D"; FAINT="#c2c6c2"

fig, ax = plt.subplots(figsize=(9.5, 6.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# plot region the two lines live in
x0, x1 = 2.2, 8.2          # shared x-span of both arrows
ylo, yhi = 3.2, 7.6        # low and high y of the crossing lines
# rising line: (x0,ylo) -> (x1,yhi);  falling line: (x0,yhi) -> (x1,ylo)
# they cross exactly at the midpoint:
cx, cy = (x0+x1)/2, (ylo+yhi)/2   # = (5.2, 5.4)

# --- axes ---
ax.add_patch(FancyArrowPatch((1.3,2.0),(1.3,9.2),arrowstyle="-|>",mutation_scale=16,lw=1.8,color=GREY))
ax.add_patch(FancyArrowPatch((1.3,2.0),(9.3,2.0),arrowstyle="-|>",mutation_scale=16,lw=1.8,color=GREY))
ax.text(0.75,6.0,"dimensions",rotation=90,va="center",ha="center",fontsize=14,color=GREY)
ax.text(9.3,1.55,"task demand",va="top",ha="right",fontsize=14,color=GREY)

# --- the two diverging arrows (cross exactly at cx,cy) ---
ax.add_patch(FancyArrowPatch((x0,ylo),(x1,yhi),arrowstyle="-|>",mutation_scale=20,lw=3.2,color=GREEN))
ax.add_patch(FancyArrowPatch((x0,yhi),(x1,ylo),arrowstyle="-|>",mutation_scale=20,lw=3.2,color=GREY))

# labels anchored at each arrow's head
ax.text(x1+0.15, yhi+0.35, "more demand →\nmore complex?", fontsize=13, color=GREEN,
        va="bottom", ha="right", fontweight="bold", linespacing=1.2)
ax.text(x1+0.15, ylo-0.35, "constraints\ncompress", fontsize=13, color=GREY,
        va="top", ha="right", linespacing=1.2)

# --- crossing marker + ? exactly at (cx,cy) ---
ax.plot(cx, cy, marker="X", ms=16, color=ORANGE, zorder=6,
        markeredgecolor="white", markeredgewidth=1.4)
ax.text(cx+0.45, cy, "?", fontsize=30, color=ORANGE, fontweight="bold",
        va="center", ha="left", zorder=7)

# --- passive -> active band under the x-axis ---
ax.add_patch(FancyArrowPatch((1.5,0.7),(9.0,0.7),arrowstyle="-|>",mutation_scale=13,lw=1.4,color=FAINT))
ax.text(1.5,0.25,"passive",fontsize=12,color=GREY,va="top",ha="left")
ax.text(9.0,0.25,"active",fontsize=12,color=GREY,va="top",ha="right")

plt.tight_layout()
fig.savefig("intuition_slide.png",dpi=200,transparent=True,bbox_inches="tight")
fig.savefig("intuition_slide.pdf",transparent=True,bbox_inches="tight")
print("saved intuition_slide.png + .pdf")
