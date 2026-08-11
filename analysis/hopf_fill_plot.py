#!/usr/bin/env python3
"""3-D isotropic filler vs the 2-D sheet (coil) families."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/Users/yash/knot-s3/"
h=json.load(open(R+"notes/hopf_fill_data.json")); hd=json.load(open(R+"notes/hopf_dimension.json"))
G=h["Gamma"]; c0,A=h["c0"],h["A"]
L=np.array(h["L"]); E=np.array(h["E_q"])
tre=json.load(open(R+"notes/trefoil_Lsweep_pipeline.json"))
f8=json.load(open(R+"notes/fig8_Lsweep.json"))
def coil(d):
    rs=[r for r in d["rows"] if r["seg"] in("ampramp","freqramp")]
    u={round(r["L"],9):r for r in rs}; rs=sorted(u.values(),key=lambda r:r["L"])
    return np.array([r["L"] for r in rs]),np.array([r["E_q"] for r in rs])
Lt,Et=coil(tre); Lf,Ef=coil(f8)

fig=plt.figure(figsize=(14.5,5.2))
gs=fig.add_gridspec(1,3,width_ratios=[1.5,1,1],wspace=.28)
ax=fig.add_subplot(gs[0,0])
x=np.logspace(np.log10(120),5,200)
ax.axhline(G,color="tab:red",ls=":",lw=1.6,label=r"$\Gamma=\frac{2}{\pi}\mathrm{Si}(2\pi)=0.9028$")
ax.plot(x,c0-A/np.sqrt(x),"-",c="tab:purple",lw=1.3,alpha=.6)
ax.plot(L,E,"o",c="tab:purple",ms=7,label=r"3-D isotropic (Hopf):  $\Gamma-2.71\,L^{-1/2}$")
ax.plot(Lt,Et,"s",c="tab:blue",ms=4,label=r"trefoil coil (sheet):  $0.372\ln L$")
ax.plot(Lf,Ef,"^",c="tab:red",ms=4,label=r"fig-8 coil (sheet):  $0.650\ln L$")
ax.plot(x,0.3719*np.log(x)-0.7167,"--",c="tab:blue",lw=1.0)
ax.plot(x,0.6504*np.log(x)-1.5012,"--",c="tab:red",lw=1.0)
ax.set_xscale("log"); ax.set_xlim(15,1.2e5); ax.set_ylim(0,6.2)
ax.set_xlabel("$L$"); ax.set_ylabel(r"$E_q$")
ax.set_title("3-D filling stays BELOW $\\Gamma$ forever;\nsheets cross it and diverge",fontsize=11)
ax.grid(alpha=.25,which="both"); ax.legend(fontsize=8.2,loc="upper left")

a2=fig.add_subplot(gs[0,1])
a2.plot(L,(G-E)*np.sqrt(L),"o-",c="tab:purple",ms=6)
a2.axhline(((G-E)*np.sqrt(L)).mean(),ls="--",c="0.5")
a2.set_xscale("log"); a2.set_xlabel("$L$"); a2.set_ylabel(r"$(\Gamma-E_q)\sqrt{L}$")
a2.set_ylim(2.4,3.1)
a2.set_title(r"deficit is exactly $L^{-1/2}$"+"\n(flat to 1.4%% over 14x in $L$)",fontsize=10.5)
a2.grid(alpha=.25,which="both")

a3=fig.add_subplot(gs[0,2])
r=np.array(hd["r"]); m=np.array(hd["m"])
a3.plot(r,m,"-",c="tab:purple",lw=1.8,label="3-D Hopf (slope 2.93)")
ld=json.load(open(R+"notes/local_dimension.json"))
for o,c in zip(ld,("tab:blue","tab:red")):
    a3.plot(o["r"],o["m"],"-",c=c,lw=1.3,
            label="%s coil (slope %.2f)"%(o["tag"].split()[0].title(),o["dim"]))
xx=np.logspace(-1.4,-0.2,10)
for p,st in ((2,"--"),(3,":")):
    a3.plot(xx,1.5*(xx/xx[0])**p,st,c="0.5",lw=1.2,label="slope %d"%p)
a3.set_xscale("log"); a3.set_yscale("log"); a3.set_xlim(8e-3,1)
a3.set_xlabel("$r$"); a3.set_ylabel("$m(r)$")
a3.set_title("mass-radius: 3 vs 2",fontsize=10.5)
a3.legend(fontsize=7.6,loc="lower right"); a3.grid(alpha=.25,which="both")
fig.savefig(R+"notes/hopf_vs_sheet.png",dpi=150,bbox_inches="tight")
print("→ notes/hopf_vs_sheet.png")
