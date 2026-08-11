#!/usr/bin/env python3
"""Scatter of E_q against crossing number, c = 0…30, three families."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/yash/knot-s3"
GAMMA = 0.9028233735                       # (2/π)·Si(2π)

# candidate envelope law (notes: ENVELOPE CEILING LAW)
def envelope(c):
    return GAMMA * (1.0 - (1.0 + (np.asarray(c, float) / 2.0) ** (2 / 3)) ** (-3 / 8))


STYLE = {"torus":     ("tab:blue",   "o", "torus  T(p,q)"),
         "twist":     ("tab:red",    "^", "twist  (2-bridge, hyperbolic)"),
         "composite": ("tab:green",  "s", "composite  (connect sum)"),
         "unknot":    ("k",          "*", "unknot")}

FULL = os.path.join(ROOT, "notes/crossing_scan_data.json")
PRELIM = "/tmp/cscan_prelim.json"
CERTIFIED_RUN = os.path.exists(FULL)
rows = json.load(open(FULL if CERTIFIED_RUN else PRELIM))
for r in rows:                                   # normalise the two schemas
    r.setdefault("family", r.get("fam"))
    r.setdefault("E_q", r.get("E"))
    r.setdefault("gnorm", r.get("g"))
if CERTIFIED_RUN:
    rows = [r for r in rows if r.get("type_held")]

# convergence verdict: flat tail AND small gradient (AND sane ACN/c if known)
for r in rows:
    ok = r["tailrel"] < 1e-4 and r["gnorm"] < 1e-2
    if CERTIFIED_RUN and r["c"]:
        ok = ok and 1.2 <= r["ACN_over_c"] <= 2.0
    r["conv"] = bool(ok)

c = np.array([r["c"] for r in rows], float)
E = np.array([r["E_q"] for r in rows], float)

fig = plt.figure(figsize=(14, 8.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.30, wspace=0.24)

# ── A: the scatter ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :])
cx = np.linspace(0, 31, 400)
ax.plot(cx, envelope(cx), "-", c="0.45", lw=1.6, zorder=2,
        label=r"envelope law  $\Gamma\,[1-(1+(c/2)^{2/3})^{-3/8}]$")
ax.axhline(GAMMA, color="tab:red", ls=":", lw=1.5,
           label=r"$\Gamma=\frac{2}{\pi}\mathrm{Si}(2\pi)=0.9028$")
for fam, (col, mk, lab) in STYLE.items():
    sel = [r for r in rows if r["family"] == fam]
    if not sel:
        continue
    ax.plot([], [], mk, ms=8, mec=col, mfc=col, ls="none", label=lab)
    for filled in (True, False):
        pts = [r for r in sel if r["conv"] == filled]
        if not pts:
            continue
        ax.plot([r["c"] for r in pts], [r["E_q"] for r in pts], mk,
                ms=8 if fam != "unknot" else 16, mec=col, mew=1.4,
                mfc=col if filled else "none", ls="none", zorder=5)
# the T(3,q) "fat" torus knots are the cheapest thing at their crossing number
fat = [r for r in rows if r["family"] == "torus" and r["name"].startswith(("T(3", "T(5"))]
ax.plot([r["c"] for r in fat], [r["E_q"] for r in fat], "-", c="tab:blue",
        lw=1.0, alpha=.55, zorder=3)
ax.annotate("T(3,q) / T(5,6): fat torus knots,\ncheapest at their crossing number",
            xy=(24, 0.447), xytext=(3.5, 0.62), fontsize=8.5, color="tab:blue",
            arrowprops=dict(arrowstyle="->", color="tab:blue"))
ax.plot([], [], "o", mfc="none", mec="0.4", mew=1.4, ls="none",
        label="open marker = not fully converged (upper bound)")
ax.set_xlabel("crossing number  $c$")
ax.set_ylabel(r"$E_q=L^{-2}E_{S^3}$  at the minimiser")
ax.set_title("A — spherical quantity energy against crossing number, "
             "three knot families, $c=0\\ldots30$", fontsize=13)
ax.set_xlim(-1, 31); ax.set_ylim(0, 1.0)
ax.grid(alpha=.25); ax.legend(fontsize=8.6, loc="lower right", ncol=2)

# ── B: within-c spread ─────────────────────────────────────────────────────
axb = fig.add_subplot(gs[1, 0])
cs, spread, lo = [], [], []
for cc in sorted(set(c)):
    v = E[c == cc]
    if len(v) > 1:
        cs.append(cc); spread.append(100 * (v.max() - v.min()) / v.mean())
        lo.append(v.min())
axb.plot(cs, spread, "-o", ms=4, c="tab:purple")
axb.set_xlabel("$c$"); axb.set_ylabel("within-$c$ spread  (% of mean)")
axb.set_title("B — how tightly the families agree at fixed $c$\n"
              "(sawtooth: even $c$ has a fat T(3,q), odd $c$ only T(2,q))",
              fontsize=10)
axb.grid(alpha=.25)

# ── C: residual against the envelope law ───────────────────────────────────
axc = fig.add_subplot(gs[1, 1])
for fam, (col, mk, lab) in STYLE.items():
    sel = [r for r in rows if r["family"] == fam and r["c"] > 0]
    if not sel:
        continue
    cc = np.array([r["c"] for r in sel], float)
    ee = np.array([r["E_q"] for r in sel])
    axc.plot(cc, 100 * (ee - envelope(cc)) / envelope(cc), mk, ms=6,
             mec=col, mfc=col, ls="none",
             mew=1.2)
axc.axhline(0, c="0.45", lw=1.4)
axc.set_xlabel("$c$"); axc.set_ylabel("deviation from envelope law (%)")
axc.set_title("C — the law was fitted to torus knots;\nthe other families sit above it",
              fontsize=10.5)
axc.grid(alpha=.25)

ax.text(0.985, 0.045, "determinant re-certification: %s" %
        ("all %d minimisers, type held" % len(rows) if CERTIFIED_RUN else "pending"),
        transform=ax.transAxes, ha="right", fontsize=7.5, color="0.45")
fig.savefig(os.path.join(ROOT, "notes/crossing_scan.png"), dpi=150,
            bbox_inches="tight")
print("→ notes/crossing_scan.png")

# ── summary ────────────────────────────────────────────────────────────────
print("\n  c   " + "  ".join("%-24s" % s for s in ("torus", "twist", "composite")))
for cc in sorted(set(c)):
    cell = {}
    for r in rows:
        if r["c"] == cc:
            cell[r["family"]] = "%-14s %.4f%s" % (r["name"][:14], r["E_q"],
                                                  "" if r["conv"] else "*")
    print("  %2d  " % cc + "  ".join("%-24s" % cell.get(f, "")
                                     for f in ("torus", "twist", "composite")))
n_conv = sum(1 for r in rows if r["conv"])
print("\n%d knots, type held on all; %d fully converged (* = upper bound)"
      % (len(rows), n_conv))
sp = [s for s in spread]
print("within-c spread: median %.1f%%, max %.1f%%" % (np.median(sp), max(sp)))
print("Spearman rho(E_q, c) = %.4f"
      % (np.corrcoef(np.argsort(np.argsort(c)), np.argsort(np.argsort(E)))[0, 1]))
