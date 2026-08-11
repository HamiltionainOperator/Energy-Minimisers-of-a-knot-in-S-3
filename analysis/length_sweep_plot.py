#!/usr/bin/env python3
"""Plot the fixed-trefoil E_q(L) sweep on S³, L: 0 → ∞.

Reads the build manifest + the `energy_s3 --eval` DUAL lines and writes
notes/trefoil_Lsweep_pipeline.{png,json}.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/yash/knot-s3"
SW = os.environ.get("LSWEEP_DIR", os.path.join(ROOT, "output/Lsweep"))
NOTES = os.path.join(ROOT, "notes")
FIGNAME = os.environ.get("LSWEEP_FIG", "trefoil_Lsweep_pipeline")
GAMMA = 0.9028233735          # (2/π)·Si(2π) — envelope of the minima
A_STAR = float(os.environ.get("LSWEEP_A", 0.18))   # coil amplitude in ℝ³


def tube_area_s3(m_phi=64, m_s=8000):
    """S³ area of the cylinder the coil winds on.  The coil is a SURFACE-filling
    family: all its length sits on the boundary of the tube of radius A_STAR
    around the base knot, so its strand spacing is (area)/L and the asymptotic
    log-slope should be 2π/area — the same law that gives 1/π on the Clifford
    torus (area 2π²).  The area must be measured in S³, i.e. weighted by the
    conformal factor λ = 2/(1+|y|²) of the inverse stereographic lift."""
    import importlib.util
    sp = importlib.util.spec_from_file_location(
        "lsb", os.path.join(ROOT, "analysis/length_sweep_build.py"))
    lsb = importlib.util.module_from_spec(sp); sp.loader.exec_module(lsb)
    P = lsb.resample_periodic(lsb.read_vect(lsb.BASE), m_s)
    U, V, _ = lsb.closed_rmf(P)
    area = 0.0
    for phi in np.linspace(0, 2 * np.pi, m_phi, endpoint=False):
        Q = P + A_STAR * (np.cos(phi) * U + np.sin(phi) * V)
        lam = 2.0 / (1.0 + np.einsum("ij,ij->i", Q, Q))
        dl = np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1)
        area += (A_STAR * 2 * np.pi / m_phi) * (lam**2 * dl).sum()
    return float(area)


AREA = tube_area_s3()                       # S³ area of the coil's cylinder
SLOPE_PRED = 2 * np.pi / AREA               # surface-filling slope, dE_q/dlnL
CERTIFIED = os.environ.get("LSWEEP_CERT", "G2_k0016,G2_k0032,G2_k0064").split(",")
L_TARGET = 1e5                # how far the extrapolation is carried


def parse_eval():
    """Join `energy_s3 --eval` output (DUAL lines) with the build manifest."""
    man = {m["tag"]: m for m in json.load(open(os.path.join(SW, "manifest.json")))}
    out = []
    for line in open(os.path.join(SW, "eval.txt")):
        if not line.startswith("DUAL"):
            continue
        f = line.split()
        tag = os.path.basename(f[1])[:-5]
        if tag not in man:
            continue
        d = dict(tag=tag)
        for kv in f[2:]:
            k, v = kv.split("=")
            d[k] = float(v)
        d.update({k: man[tag][k] for k in ("seg", "param", "L_r3", "gap", "edge")})
        d["n"] = man[tag]["n"]
        out.append(d)
    json.dump(out, open(os.path.join(SW, "sweep.json"), "w"), indent=1)
    return out


rows = parse_eval()
by = {}
for r in rows:
    by.setdefault(r["seg"], []).append(r)


def dedupe(rs):
    seen, out = set(), []
    for r in sorted(rs, key=lambda r: r["L"]):
        key = round(r["L"], 9)
        if key not in seen:
            seen.add(key); out.append(r)
    return out


shrink = dedupe(by["shrink"])                       # ℝ³ scale s ≤ 1
scaleup = dedupe(by["scaleup"])                     # ℝ³ scale s > 1
base = min(rows, key=lambda r: r["E_q"])
coil = [c for c in dedupe(by["ampramp"] + by["freqramp"])
        if c["L"] > base["L"] * 1.000001]

main = shrink + coil                                # the single 0→∞ axis
main_L = np.array([r["L"] for r in main])
main_E = np.array([r["E_q"] for r in main])
o = np.argsort(main_L); main_L, main_E = main_L[o], main_E[o]

Estar = float(np.median([r["E_geo"] for r in shrink if r["L"] < 0.1]))

cl = np.array([r["L"] for r in coil]); ce = np.array([r["E_q"] for r in coil])
i = int(np.argmax(ce > GAMMA))
L_gamma = float(np.exp(np.interp(np.log(GAMMA), np.log(ce[i - 1:i + 1]),
                                 np.log(cl[i - 1:i + 1]))))

# log law on the last decade of measured coil data
fit = cl > cl.max() / 10.0
slope, icept = np.polyfit(np.log(cl[fit]), ce[fit], 1)
# local slope, for the drift towards the surface-filling prediction
loc_L = np.sqrt(cl[1:] * cl[:-1])
loc_s = np.diff(ce) / np.diff(np.log(cl))
keep = loc_L > 22.0            # drop the amplitude-ramp segment (not yet a coil)
loc_L, loc_s = loc_L[keep], loc_s[keep]

# two extrapolations to L_TARGET: current fitted slope, and the asymptotic one
Lx = np.logspace(np.log10(cl.max()), np.log10(L_TARGET), 40)
ex_hi = slope * np.log(Lx) + icept
ex_lo = ce[-1] + SLOPE_PRED * (np.log(Lx) - np.log(cl[-1]))

fig = plt.figure(figsize=(15, 9.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], hspace=0.34, wspace=0.29)

# ── A: the whole sweep, 0 → 1e5 ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :2])
Lw = np.logspace(-4.5, 1.05, 80)
ax.plot(Lw, Estar / Lw**2, "--", c="0.5", lw=1.3,
        label=r"collapse wall  $E_q=E^\star/L^2$,  $E^\star=%.2f$" % Estar)
ax.axhline(GAMMA, color="tab:red", ls=":", lw=1.5,
           label=r"$\Gamma=\frac{2}{\pi}\mathrm{Si}(2\pi)=0.9028$")
ax.plot(main_L, main_E, "-", c="tab:blue", lw=2.0, zorder=4,
        label=os.environ.get("LSWEEP_LABEL", "fixed trefoil") + r", measured  ($L$: %.0e … %.0f)" % (main_L.min(), main_L.max()))
ax.plot([r["L"] for r in scaleup], [r["E_q"] for r in scaleup], "-",
        c="tab:orange", lw=1.6, alpha=.9, zorder=3,
        label=r"$\mathbb{R}^3$ scale-up wing ($s>1$)")
ax.plot(cl, ce, "s", ms=3.6, c="tab:green", zorder=5, label="coil branch")
ax.fill_between(Lx, ex_lo, ex_hi, color="tab:green", alpha=.20, zorder=2)
ax.plot(Lx, ex_hi, "--", c="tab:green", lw=1.4, zorder=3,
        label=r"extrapolation to $10^5$ (band: fitted → asymptotic slope)")
ax.plot(Lx, ex_lo, "--", c="tab:green", lw=1.0, zorder=3)
ax.plot([base["L"]], [base["E_q"]], "*", ms=19, c="crimson", zorder=6,
        label=r"minimum $%.6f$" % base["E_q"])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1.5e-4, 4e5); ax.set_ylim(0.1, 3e9)
ax.set_xlabel(r"S³ length  $L$"); ax.set_ylabel(r"$E_q=L^{-2}E_{S^3}$")
ax.set_title(os.environ.get("LSWEEP_TITLE", r"A fixed knot type (trefoil) swept over length $L:0\to\infty$ on $S^3$"),
             fontsize=13)
ax.grid(alpha=.25, which="both"); ax.legend(fontsize=8.4, loc="upper right")
ax.annotate("collapse: knot shrinks into a ball\n(either pole) — slope exactly $-2$",
            xy=(6e-4, 3e6), color="0.35", fontsize=9)
ax.annotate("coiling", xy=(2e3, 8), color="tab:green", fontsize=10)
ax.add_patch(plt.Rectangle((9, 0.2), 400 - 9, 2.0, fill=False, ec="k",
                           ls="--", lw=1.0, zorder=8))
ax.text(11, 3.2, "panel B", fontsize=8)

# ── B: valley zoom ─────────────────────────────────────────────────────────
axb = fig.add_subplot(gs[0, 2])
axb.plot([r["L"] for r in shrink], [r["E_q"] for r in shrink], "-o", ms=3,
         c="tab:blue", label=r"scale down $s\!<\!1$")
axb.plot([r["L"] for r in scaleup], [r["E_q"] for r in scaleup], "-o", ms=3,
         c="tab:orange", label=r"scale up $s\!>\!1$")
axb.plot(cl, ce, "-s", ms=4, c="tab:green", label="coil")
axb.axhline(GAMMA, color="tab:red", ls=":", lw=1.5)
axb.plot([base["L"]], [base["E_q"]], "*", ms=19, c="crimson", zorder=6)
for tag in CERTIFIED:
    r = next(x for x in rows if x["tag"] == tag)
    axb.plot([r["L"]], [r["E_q"]], "o", mfc="none", mec="k", ms=11, mew=1.3)
axb.plot([], [], "o", mfc="none", mec="k", ms=9, mew=1.3, label="det = 3 re-checked")
axb.axvline(base["L"], color="0.8", lw=1)
axb.text(base["L"] * 1.05, 1.86, r"$L_{\max}$ (fold)", fontsize=7.5,
         rotation=90, va="top", color="0.45")
axb.text(10.5, GAMMA * 1.05, r"$\Gamma$", color="tab:red", fontsize=10)
axb.annotate(r"$\Gamma$ at $L\approx%.0f$" % L_gamma, xy=(L_gamma, GAMMA),
             xytext=(28, 0.45), fontsize=8.5, color="tab:red",
             arrowprops=dict(arrowstyle="->", color="tab:red"))
axb.set_xscale("log"); axb.set_xlim(9, 420); axb.set_ylim(0.2, 1.9)
axb.set_xlabel(r"$L$"); axb.set_ylabel(r"$E_q$")
axb.set_title("B — the valley and the climb past $\\Gamma$", fontsize=10.5)
axb.grid(alpha=.25, which="both"); axb.legend(fontsize=7.8, loc="upper left")

# ── C: E_S3 identity ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(main_L, main_E * main_L**2, "-o", ms=2.5, c="tab:blue")
ax2.axhline(Estar, ls="--", c="0.55")
ax2.text(2e-3, Estar * 1.9, r"$E^\star=%.2f$" % Estar, color="0.35", fontsize=9)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel(r"$L$"); ax2.set_ylabel(r"$E_{S^3}=L^2E_q$")
ax2.set_title(r"C — $E_{S^3}$ pinned at $E^\star$ over 5 decades of $L$"
              "\n(the $1/L^2$ wall is an identity)", fontsize=10)
ax2.grid(alpha=.25, which="both")

# ── D: the coil branch, lin-y, out to 1e5 ──────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(cl, ce, "s", ms=4.5, c="tab:green", label="measured")
ax3.fill_between(Lx, ex_lo, ex_hi, color="tab:green", alpha=.20)
ax3.plot(Lx, ex_hi, "--", c="k", lw=1.2,
         label=r"$%.4f\ln L%+.4f$ (fit, last decade)" % (slope, icept))
ax3.plot(Lx, ex_lo, "--", c="tab:brown", lw=1.2,
         label=r"surface-filling slope $2\pi/A_{S^3}=%.4f$" % SLOPE_PRED)
ax3.axhline(GAMMA, color="tab:red", ls=":", lw=1.5)
ax3.text(3e4, GAMMA * 1.06, r"$\Gamma$", color="tab:red", fontsize=9)
ax3.set_xscale("log"); ax3.set_xlim(9, 2e5)
ax3.set_xlabel(r"$L$"); ax3.set_ylabel(r"$E_q$")
ax3.set_title(r"D — the $L\to\infty$ branch grows like $\ln L$;"
              "\n$E_q(10^5)\\approx%.2f$–$%.2f$" % (ex_lo[-1], ex_hi[-1]), fontsize=10)
ax3.grid(alpha=.25, which="both"); ax3.legend(fontsize=7.6, loc="upper left")

# ── E: local slope drift on the coil branch ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(loc_L, loc_s, "-o", ms=4, c="tab:purple", label=r"local $dE_q/d\ln L$")
ax4.axhline(SLOPE_PRED, ls="--", c="tab:brown",
            label=r"$2\pi/A_{S^3}=%.4f$  predicted" % SLOPE_PRED)
ax4.axhline(1 / np.pi, ls=":", c="0.45", label=r"$1/\pi$ (Clifford, area $2\pi^2$)")
ax4.set_xscale("log")
lo, hi = min(loc_s.min(), SLOPE_PRED), max(loc_s.max(), SLOPE_PRED)
pad = 0.12 * (hi - lo) + 0.01
ax4.set_ylim(lo - pad, hi + pad)
ax4.set_xlabel(r"$L$"); ax4.set_ylabel(r"$dE_q/d\ln L$")
_dev = 100 * (loc_s[-1] - SLOPE_PRED) / SLOPE_PRED
ax4.set_title("E — measured log-slope %.4f vs predicted\n$2\\pi/A_{S^3}=%.4f$  (%+.0f%%)"
              % (loc_s[-1], SLOPE_PRED, _dev), fontsize=10)
ax4.grid(alpha=.25, which="both"); ax4.legend(fontsize=7.6, loc="upper right")

fig.savefig(os.path.join(NOTES, FIGNAME + ".png"), dpi=150,
            bbox_inches="tight")
print("→ notes/%s.png" % FIGNAME)

json.dump(dict(
    base=dict(tag=base["tag"], E_q=base["E_q"], L=base["L"], E_S3=base["E_geo"]),
    E_star=Estar, Gamma=GAMMA, L_gamma_crossing=L_gamma,
    log_fit=dict(slope=float(slope), intercept=float(icept),
                 L_min_fit=float(cl[fit].min())),
    slope_pred_surface_filling=float(SLOPE_PRED), coil_cylinder_area=float(AREA),
    L_min=float(main_L.min()), L_max=float(main_L.max()),
    E_q_at_Lmin=float(main_E[np.argmin(main_L)]),
    E_q_at_Lmax=float(main_E[np.argmax(main_L)]),
    extrapolated_E_q_at_1e5=[float(ex_lo[-1]), float(ex_hi[-1])],
    L_max_scaling_fold=float(max(r["L"] for r in shrink + scaleup)),
    certified=CERTIFIED, rows=rows),
    open(os.path.join(NOTES, FIGNAME + ".json"), "w"), indent=1)
print("→ notes/%s.json" % FIGNAME)
print("E*=%.4f  Γ@L=%.2f" % (Estar, L_gamma))
print("coil fit (L>%.0f): E_q = %.4f lnL %+.4f   resid max %.4f"
      % (cl[fit].min(), slope, icept,
         np.abs(ce[fit] - (slope * np.log(cl[fit]) + icept)).max()))
print("L: %.3e … %.2f   E_q: %.6f … %.4e"
      % (main_L.min(), main_L.max(), main_E.min(), main_E.max()))
print("local slope tail:", " ".join("%.0f:%.3f" % (l, s)
                                    for l, s in zip(loc_L[-6:], loc_s[-6:])))
print("E_q(1e5) extrapolated: %.3f (asymptotic slope) … %.3f (fitted slope)"
      % (ex_lo[-1], ex_hi[-1]))
