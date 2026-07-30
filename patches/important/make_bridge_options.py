"""Three layout options for the bridge slide (sensation/action -> visuomotor
control). Causal chain: task demand -> constraints -> coupling -> low-dim manifold.
Outputs bridge_opt1/2/3.png (transparent). Deck palette."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse, Polygon

GREEN="#2f5233"; GREY="#6b6f6b"; ORANGE="#E8A33D"; LGREEN="#d8e4d8"; FAINT="#c2c6c2"

def box(ax,x,y,w,h,text,fc=LGREEN,ec=GREEN,fs=12,bold=False,tc="#1a1a1a"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc,ec=ec,lw=1.6,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,zorder=3,
            color=tc,fontweight="bold" if bold else "normal",linespacing=1.15)
def arr(ax,x0,y0,x1,y1,c=GREY,lw=2.4,ms=18):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=ms,lw=lw,color=c,zorder=1))

# ============ OPTION 1: horizontal flow of boxes -> manifold ============
fig,ax=plt.subplots(figsize=(12,3.6)); ax.set_xlim(0,12); ax.set_ylim(0,3.6); ax.axis("off")
box(ax,0.2,1.2,2.3,1.2,"task demands\n↑",LGREEN,GREEN,12,True)
arr(ax,2.6,1.8,3.2,1.8)
box(ax,3.3,1.2,2.5,1.2,"salient info\n+ consequences",LGREEN,GREEN,11.5)
arr(ax,5.9,1.8,6.5,1.8)
box(ax,6.6,1.2,2.3,1.2,"action space\nconstrained",LGREEN,GREEN,11.5)
arr(ax,9.0,1.8,9.6,1.8)
# manifold = small tight ellipse
ax.add_patch(Ellipse((10.8,1.8),1.5,0.6,angle=-18,fc=GREEN,ec=GREEN,alpha=0.85,zorder=2))
ax.text(10.8,0.95,"coupled,\nlow-dimensional",ha="center",va="top",fontsize=11,color=GREEN,fontweight="bold",linespacing=1.15)
plt.tight_layout(); fig.savefig("bridge_opt1.png",dpi=200,transparent=True,bbox_inches="tight"); plt.close()

# ============ OPTION 2: funnel squeezing scatter -> manifold ============
fig,ax=plt.subplots(figsize=(11,5.2)); ax.set_xlim(0,11); ax.set_ylim(0,5.2); ax.axis("off")
# funnel (trapezoid, wide left -> narrow right)
ax.add_patch(Polygon([(1.2,0.6),(1.2,4.6),(8.6,3.1),(8.6,2.1)],closed=True,fc="#e9ece9",ec=GREY,lw=1.4,zorder=1))
ax.text(1.6,4.9,"task demands constrain the action space",fontsize=13,color=GREY,ha="left",va="bottom")
# wide scattered cloud (high-dim) at the mouth
rng=np.random.default_rng(1)
sx=2.4+rng.normal(0,0.55,120); sy=2.6+rng.normal(0,1.1,120)
ax.scatter(sx,sy,s=10,color=GREY,alpha=0.55,zorder=2)
ax.text(2.4,0.75,"independent\nchannels",fontsize=11,color=GREY,ha="center",va="top")
# narrow coupled manifold at the spout
ax.add_patch(Ellipse((7.7,2.6),1.4,0.42,angle=-8,fc=GREEN,ec=GREEN,alpha=0.85,zorder=3))
ax.text(9.0,2.6,"coupled,\nlow-dim\nmanifold",fontsize=12,color=GREEN,fontweight="bold",ha="left",va="center",linespacing=1.15)
plt.tight_layout(); fig.savefig("bridge_opt2.png",dpi=200,transparent=True,bbox_inches="tight"); plt.close()

# ============ OPTION 3: scatter cloud vs coupled cloud ============
fig,ax=plt.subplots(figsize=(11,4.6)); ax.set_xlim(0,11); ax.set_ylim(0,4.6); ax.axis("off")
rng=np.random.default_rng(3)
# left: scattered (high-dim, independent)
lx=2.0+rng.normal(0,0.7,140); ly=2.5+rng.normal(0,0.7,140)
ax.scatter(lx,ly,s=12,color=GREY,alpha=0.55,zorder=2)
ax.text(2.0,0.55,"independent channels\n(high-dimensional)",fontsize=11.5,color=GREY,ha="center",va="top",linespacing=1.15)
# right: coupled along one axis (low-dim)
t=rng.normal(0,1.0,140); rx=8.2+0.85*t+rng.normal(0,0.12,140); ry=2.5+0.5*t+rng.normal(0,0.12,140)
ax.scatter(rx,ry,s=12,color=GREEN,alpha=0.6,zorder=2)
ax.text(8.2,0.55,"coupled sensorimotor\n(low-dimensional)",fontsize=11.5,color=GREEN,fontweight="bold",ha="center",va="top",linespacing=1.15)
# arrow between
arr(ax,4.0,2.5,6.4,2.5,GREEN,3.0,22)
ax.text(5.2,3.0,"task demand\ncouples channels",fontsize=12,color=GREEN,ha="center",va="bottom",fontweight="bold",linespacing=1.15)
plt.tight_layout(); fig.savefig("bridge_opt3.png",dpi=200,transparent=True,bbox_inches="tight"); plt.close()

print("saved bridge_opt1.png, bridge_opt2.png, bridge_opt3.png")
