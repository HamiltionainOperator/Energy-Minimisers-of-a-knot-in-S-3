#!/usr/bin/env python3
"""E_{S^3}/L^2 against L for a FIXED knot type whose length is grown uniformly
(the coil family: every part of the knot is wound equally, so length is added
evenly along the curve rather than in one place)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/Users/yash/knot-s3/"
G = 0.9028233735

def coil(path):
    d = json.load(open(R + path))
    rs = [r for r in d["rows"] if r["seg"] in ("ampramp", "freqramp")]
    u = {round(r["L"], 9): r for r in rs}
    rs = sorted(u.values(), key=lambda r: r["L"])
    return (np.array([r["L"] for r in rs]),
            np.array([r["E_geo"] for r in rs]),
            np.array([r["E_q"] for r in rs]))

sets = [("trefoil $3_1$", "notes/trefoil_Lsweep_pipeline.json", "tab:blue", "o"),
        ("figure-eight $4_1$", "notes/fig8_Lsweep.json", "tab:red", "^")]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.1))
print("  E_{S^3}/L^2  for a fixed knot, length grown uniformly\n")
for name, path, c, mk in sets:
    L, Eg, Eq = coil(path)
    a1.plot(L, Eg, mk + "-", c=c, ms=4, lw=1.1, label=name)
    a2.plot(L, Eq, mk + "-", c=c, ms=5, lw=1.2, label=name)
    fit = L > L.max() / 10
    s, b = np.polyfit(np.log(L[fit]), Eq[fit], 1)
    xs = np.logspace(np.log10(L[fit].min()), np.log10(L.max() * 1.05), 30)
    a2.plot(xs, s * np.log(xs) + b, "--", c=c, lw=1.0)
    res = np.abs(Eq[fit] - (s * np.log(L[fit]) + b)).max()
    print("  %-20s L: %.1f -> %.1f    E_S3/L^2: %.4f -> %.4f" %
          (name.replace("$", ""), L[0], L[-1], Eq[0], Eq[-1]))
    print("      fit  E_S3/L^2 = %.4f ln L %+.4f   max resid %.5f" % (s, b, res))
    lo = np.diff(Eq) / np.diff(np.log(L))
    print("      local slope over the last 5 points: " +
          " ".join("%.4f" % x for x in lo[-5:]) + "\n")

for a, ttl, yl in ((a1, r"$E_{S^3}$ itself  (grows like $L^2\ln L$)", r"$E_{S^3}$"),
                   (a2, r"the ratio  $E_{S^3}/L^2$  — grows like $\ln L$", r"$E_{S^3}/L^2$")):
    a.set_xscale("log"); a.set_xlabel("S³ length  $L$"); a.set_ylabel(yl)
    a.set_title(ttl, fontsize=11); a.grid(alpha=.25, which="both"); a.legend(fontsize=9)
a1.set_yscale("log")
a2.axhline(G, color="k", ls=":", lw=1.3)
a2.text(20, G * 1.03, r"$\Gamma=0.9028$", fontsize=8.5)
fig.suptitle("Fixed knot type, length grown uniformly:  $E_{S^3}/L^2$ diverges logarithmically",
             fontsize=12.5)
fig.tight_layout()
fig.savefig(R + "notes/ratio_EL2.png", dpi=150, bbox_inches="tight")
print("  -> notes/ratio_EL2.png")
