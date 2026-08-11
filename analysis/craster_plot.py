#!/usr/bin/env python3
"""Unknotted 3-D raster vs isotropic Hopf filler vs the 2-D sheet coils."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/Users/yash/knot-s3/"
G=0.9028233735
cr=json.load(open(R+"notes/craster_data.json")); hp=json.load(open(R+"notes/hopf_fill_data.json"))
crd=json.load(open(R+"notes/craster_dim.json")); hpd=json.load(open(R+"notes/hopf_dimension.json"))
def coil(fn):
    d=json.load(open(R+"notes/"+fn)); rs=[r for r in d["rows"] if r["seg"] in("ampramp","freqramp")]
    u={round(r["L"],9):r for r in rs}; rs=sorted(u.values(),key=lambda r:r["L"])
    return np.array([r["L"] for r in rs]),np.array([r["E_q"] for r in rs])
Lt,Et=coil("trefoil_Lsweep_pipeline.json"); Lf,Ef=coil("fig8_Lsweep.json")
Lc,Ec=np.array(cr["L"]),np.array(cr["E_q"]); Lh,Eh=np.array(hp["L"]),np.array(hp["E_q"])

fig=plt.figure(figsize=(14.5,5.2)); gs=fig.add_gridspec(1,3,width_ratios=[1.55,1,1],wspace=.29)
ax=fig.add_subplot(gs[0,0]); x=np.logspace(np.log10(90),5,200)
ax.axhline(G,color="tab:red",ls=":",lw=1.6,label=r"$\Gamma=0.9028$")
ax.plot(x,cr["c0"]-cr["A"]*x**-cr["beta"],"-",c="tab:green",lw=1.2,alpha=.7)
ax.plot(Lc,Ec,"D",c="tab:green",ms=7,
        label=r"UNKNOT, 3-D raster:  $%.3f-%.2fL^{-%.2f}$"%(cr["c0"],cr["A"],cr["beta"]))
ax.plot(Lh,Eh,"o",c="tab:purple",ms=6,label=r"3-D Hopf (knotted):  $\Gamma-2.71L^{-1/2}$")
ax.plot(Lt,Et,"s",c="tab:blue",ms=3.5,label=r"trefoil coil (sheet):  $0.372\ln L$")
ax.plot(Lf,Ef,"^",c="tab:red",ms=3.5,label=r"fig-8 coil (sheet):  $0.650\ln L$")
ax.plot(x,0.3719*np.log(x)-0.7167,"--",c="tab:blue",lw=.9)
ax.plot(x,0.6504*np.log(x)-1.5012,"--",c="tab:red",lw=.9)
ax.set_xscale("log"); ax.set_xlim(15,1.2e5); ax.set_ylim(0,6.2)
ax.set_xlabel("$L$"); ax.set_ylabel("$E_q$")
ax.set_title("An UNKNOTTED curve of unbounded length\nwith BOUNDED energy",fontsize=11.5)
ax.grid(alpha=.25,which="both"); ax.legend(fontsize=8,loc="upper left")

a2=fig.add_subplot(gs[0,1])
a2.semilogx(Lc,np.diff(np.concatenate([[Ec[0]],Ec]))/np.diff(np.concatenate([[Lc[0]*0.999],np.log(Lc)*0+np.log(Lc)])+1e-12),alpha=0)
slc=np.diff(Ec)/np.diff(np.log(Lc)); slh=np.diff(Eh)/np.diff(np.log(Lh))
a2.semilogx(np.sqrt(Lc[1:]*Lc[:-1]),slc,"D-",c="tab:green",ms=5,label="unknot raster")
a2.semilogx(np.sqrt(Lh[1:]*Lh[:-1]),slh,"o-",c="tab:purple",ms=4,label="Hopf")
a2.axhline(0.3719,ls="--",c="tab:blue",label="trefoil coil (flat)")
a2.axhline(0.6504,ls="--",c="tab:red",label="fig-8 coil (flat)")
a2.set_xlabel("$L$"); a2.set_ylabel(r"local $dE_q/d\ln L$"); a2.set_ylim(0,0.75)
a2.set_title("sheets hold a constant slope;\n3-D families decay to 0",fontsize=10.5)
a2.grid(alpha=.25,which="both"); a2.legend(fontsize=7.6)

a3=fig.add_subplot(gs[0,2])
for d,c,lab in ((crd,"tab:green","unknot raster (2.83)"),(hpd,"tab:purple","Hopf (2.93)")):
    a3.plot(d["r"],d["m"],"-",c=c,lw=1.8,label=lab)
ld=json.load(open(R+"notes/local_dimension.json"))
for o,c in zip(ld,("tab:blue","tab:red")):
    a3.plot(o["r"],o["m"],"-",c=c,lw=1.2,label="%s coil (%.2f)"%(o["tag"].split()[0].title(),o["dim"]))
xx=np.logspace(-1.35,-0.25,10)
for p,st in ((2,"--"),(3,":")): a3.plot(xx,2.0*(xx/xx[0])**p,st,c="0.5",lw=1.2,label="slope %d"%p)
a3.set_xscale("log"); a3.set_yscale("log"); a3.set_xlim(8e-3,1)
a3.set_xlabel("$r$"); a3.set_ylabel("$m(r)$"); a3.set_title("mass-radius: 3 vs 2",fontsize=10.5)
a3.legend(fontsize=7.2,loc="lower right"); a3.grid(alpha=.25,which="both")
fig.savefig(R+"notes/unknot_spacefilling.png",dpi=150,bbox_inches="tight")
print("→ notes/unknot_spacefilling.png")
