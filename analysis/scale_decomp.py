#!/usr/bin/env python3
"""Space-partition view of the energy: contribution to E_{S^3}/L^2 from pairs
whose geodesic separation lies in each dyadic shell [delta, 2delta)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=json.load(open("/tmp/scales.json"))
fig,ax=plt.subplots(figsize=(8.4,5.2))
for (tag,(e,a,L)),c,mk in zip(d.items(),("tab:blue","tab:purple"),("o","s")):
    e=np.array(e); a=np.array(a); mid=np.sqrt(e[:-1]*e[1:])
    ax.plot(mid,a,mk+"-",c=c,ms=6,label="%s,  L=%.0f"%(tag,L))
ax.axhline(0,c="0.7",lw=.8)
ax.set_xscale("log"); ax.set_xlabel(r"spatial scale  $\delta$  (cell size)")
ax.set_ylabel(r"contribution to $E_{S^3}/L^2$ from scale $\delta$")
ax.set_title("Space partition: which scales carry the energy\n"
             r"flat $\Rightarrow$ each new scale adds a fixed amount $\Rightarrow$ $\ln L$;"
             "\n"r"decaying $\Rightarrow$ geometric sum $\Rightarrow$ bounded",fontsize=11)
ax.annotate("sheet: every scale pays the same ~0.2\nnew scales open as L grows -> +0.2 each",
            xy=(0.06,0.19),xytext=(0.004,0.245),fontsize=8.5,color="tab:blue",
            arrowprops=dict(arrowstyle="->",color="tab:blue"))
ax.annotate("3-D: small scales pay ~nothing\n(0.0001 at delta=0.01)\nsum dominated by delta~1",
            xy=(0.02,0.0006),xytext=(0.0035,0.09),fontsize=8.5,color="tab:purple",
            arrowprops=dict(arrowstyle="->",color="tab:purple"))
ax.grid(alpha=.25,which="both"); ax.legend(fontsize=9,loc="upper left")
fig.savefig("/Users/yash/knot-s3/notes/scale_decomposition.png",dpi=150,bbox_inches="tight")
print("-> notes/scale_decomposition.png")
