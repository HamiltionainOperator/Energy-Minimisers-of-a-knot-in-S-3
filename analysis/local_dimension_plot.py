import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=json.load(open("/Users/yash/knot-s3/notes/local_dimension.json"))
fig,(a1,a2)=plt.subplots(1,2,figsize=(12.5,5.0))
C={"TREFOIL coil":("tab:blue","o"),"FIGURE-EIGHT coil":("tab:red","^")}
for o in d:
    c,mk=C[o["tag"]]
    r=np.array(o["r"]); m=np.array(o["m"]); rho=o["rho_big"]; aS=o["a_s3"]
    a1.plot(r,m,"-",c=c,lw=1.6,label="%s  (slope %.2f)"%(o["tag"].split()[0].title(),o["dim"]))
    w=(r>2.5*rho)&(r<0.8*aS); a1.plot(r[w],m[w],mk,c=c,ms=6)
    a1.axvline(rho,color=c,ls=":",lw=.9); a1.axvline(aS,color=c,ls="--",lw=.9)
    L=np.array(o["L"]); rr=np.array(o["rho"])
    a2.plot(L,rr,mk+"-",c=c,ms=5,label=r"%s:  $\rho\sim L^{%.3f}$"%(o["tag"].split()[0].title(),o["alpha"]))
x=np.logspace(-2.1,-0.05,10)
for p,st,lab in [(1,":","slope 1  (single strand)"),(2,"-","slope 2  (sheet)"),(3,"--","slope 3  (3-D)")]:
    a1.plot(x,0.9*(x/x[0])**p*0.02,st,c="0.55",lw=1.2,label=lab)
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel(r"$r$  (S³ geodesic)")
a1.set_ylabel(r"$m(r)=\mathcal{H}^1(\gamma\cap B_r(x))$")
a1.set_title("mass–radius: both coils are SHEETS\n(dotted $=\\rho$, dashed $=$ S³ tube radius)",fontsize=10.5)
a1.legend(fontsize=8); a1.grid(alpha=.25,which="both"); a1.set_ylim(1e-2,3e2)
Lx=np.logspace(1.4,3.2,10)
a2.plot(Lx,0.30*(Lx/25.)**-1,"-",c="0.45",lw=1.4,label=r"$L^{-1}$  (sheet)")
a2.plot(Lx,0.30*(Lx/25.)**-0.5,"--",c="0.45",lw=1.4,label=r"$L^{-1/2}$  (3-D)")
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlabel("$L$")
a2.set_ylabel(r"$\rho$ = mean nearest non-adjacent strand")
a2.set_title("strand spacing vs length",fontsize=10.5)
a2.legend(fontsize=8.5); a2.grid(alpha=.25,which="both")
fig.tight_layout(); fig.savefig("/Users/yash/knot-s3/notes/local_dimension.png",dpi=150,bbox_inches="tight")
print("→ notes/local_dimension.png")
