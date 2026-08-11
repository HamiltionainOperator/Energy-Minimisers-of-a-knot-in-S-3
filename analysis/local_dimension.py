#!/usr/bin/env python3
"""
Local dimension of the coil families, two independent ways.

(1) Mass-radius.  For sampled x on the curve, m(x,r) = H^1(gamma ∩ B_r(x)),
    i.e. the curve length inside the S³ ball of radius r.  Fit log m vs log r
    in the window rho << r << (tube radius).  Slope 1 = single strand,
    2 = sheet, 3 = genuinely three-dimensional.

(2) Strand spacing.  rho = mean distance to the nearest NON-adjacent strand,
    as a function of L.  rho ~ L^-1 is a sheet, rho ~ L^-1/2 is 3-D.

Distances are S³ geodesic (arccos), since that is the metric the energy uses.
"Non-adjacent" = separated by more than DELTA in arc length along the curve,
which excludes the strand through x itself but keeps the neighbouring turn.
"""
import json
import os
import sys

import numpy as np

ROOT = "/Users/yash/knot-s3"
sys.path.insert(0, ROOT)
from s3_project import R3toS3                                    # noqa: E402


def read_vect(p):
    tok = open(p).read().split()
    n = int(tok[1])
    return np.array(tok[2:2 + 3 * n], float).reshape(n, 3)


def lift(P):
    X = R3toS3(P)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def arc(X):
    """S³ arc-length position of each vertex, and the Voronoi weights."""
    d = np.arccos(np.clip(np.einsum("ij,ij->i", X, np.roll(X, -1, 0)), -1, 1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]
    w = 0.5 * (d + np.roll(d, 1))          # length represented by each vertex
    return s[:-1], w, L


def _cos(A, B):
    return np.clip(A @ B.T, -1.0, 1.0)


def mass_radius(path, rgrid, ncent=160, seed=0, chunk=16):
    """accumulate m(r) in row chunks — never materialise ncent x n"""
    X = lift(read_vect(path))
    s, w, L = arc(X)
    rng = np.random.default_rng(seed)
    cen = rng.choice(len(X), size=min(ncent, len(X)), replace=False)
    tot = np.zeros(len(rgrid))
    for i in range(0, len(cen), chunk):
        D = np.arccos(_cos(X[cen[i:i + chunk]], X))
        for k, r in enumerate(rgrid):
            tot[k] += ((D <= r) * w[None, :]).sum()
    return tot / len(cen), L


def spacing(path, delta=0.30, ncent=1200, seed=0, chunk=64):
    """mean (and median) S³ distance to the nearest non-adjacent strand.
    Running minimum in row chunks — the full ncent x n matrix would be GBs."""
    X = lift(read_vect(path))
    s, w, L = arc(X)
    rng = np.random.default_rng(seed)
    cen = rng.choice(len(X), size=min(ncent, len(X)), replace=False)
    nn = np.empty(len(cen))
    for i in range(0, len(cen), chunk):
        c = cen[i:i + chunk]
        D = np.arccos(_cos(X[c], X))
        ds = np.abs(s[c][:, None] - s[None, :])
        ds = np.minimum(ds, L - ds)                 # cyclic arc separation
        D[ds < delta] = np.inf
        nn[i:i + chunk] = D.min(1)
    nn = nn[np.isfinite(nn)]
    return float(nn.mean()), float(np.median(nn)), L


def lam_mean(base, a_r3):
    """R3-arclength mean conformal factor over the tube: converts the R3 coil
    amplitude into the S³ tube radius that bounds the sheet window."""
    import importlib.util
    os.environ["LSWEEP_BASE"] = base
    sp = importlib.util.spec_from_file_location(
        "lsb", os.path.join(ROOT, "analysis/length_sweep_build.py"))
    lsb = importlib.util.module_from_spec(sp); sp.loader.exec_module(lsb)
    P = lsb.resample_periodic(lsb.read_vect(base), 4000)
    U, V, _ = lsb.closed_rmf(P)
    num = den = 0.0
    for phi in np.linspace(0, 2 * np.pi, 32, endpoint=False):
        Q = P + a_r3 * (np.cos(phi) * U + np.sin(phi) * V)
        lam = 2.0 / (1.0 + np.einsum("ij,ij->i", Q, Q))
        dl = np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1)
        num += (lam * dl).sum(); den += dl.sum()
    return num / den


def run(tag, sweep, a_r3, base, kmax_n=500000):
    print("\n" + "=" * 74)
    print("%s   (coil amplitude a = %.2f in R³)" % (tag, a_r3))
    print("=" * 74)
    man = {m["tag"]: m for m in json.load(open(os.path.join(sweep, "manifest.json")))}
    coil = sorted([m for m in man.values() if m["seg"] == "freqramp"],
                  key=lambda m: m["L_r3"])

    # ── (2) spacing vs L ───────────────────────────────────────────────────
    print("\n(2) nearest non-adjacent strand distance rho vs L")
    print("      L        rho(mean)   rho(median)     n")
    Ls, rhos = [], []
    for m in coil:
        if m["n"] > kmax_n:
            continue
        mu, med, L = spacing(m["path"])
        Ls.append(L); rhos.append(mu)
        print("  %8.2f   %.6f    %.6f   %7d" % (L, mu, med, m["n"]))
    Ls, rhos = np.array(Ls), np.array(rhos)
    sel = Ls > 3 * Ls.min()
    al, c0 = np.polyfit(np.log(Ls[sel]), np.log(rhos[sel]), 1)
    print("\n  fit rho ~ L^alpha over the top decade:  alpha = %+.4f" % al)
    print("     sheet predicts -1.0000,  3-D predicts -0.5000")

    # ── (1) mass-radius on the longest affordable coil ─────────────────────
    a_s3 = lam_mean(base, a_r3) * a_r3        # S³ tube radius
    big = [m for m in coil if m["n"] <= kmax_n][-1]
    rho_big, _med, L_big = spacing(big["path"])
    print("\n(1) mass-radius on %s (n=%d, L=%.1f, rho=%.5f)"
          % (big["tag"], big["n"], L_big, rho_big))
    rg = np.logspace(np.log10(rho_big * 0.4), np.log10(0.9), 40)
    M, L = mass_radius(big["path"], rg)
    lo = np.log(rg); lm = np.log(np.maximum(M, 1e-300))
    sl = np.gradient(lm, lo)
    print("       r        m(r)      d log m / d log r")
    for r, mm, ss in zip(rg, M, sl):
        mark = ""
        if rho_big * 2.5 < r < 0.8 * a_s3:
            mark = "   <- window rho << r << a"
        print("   %8.5f  %10.5f      %5.2f%s" % (r, mm, ss, mark))
    print("  (S³ tube radius a_S3 = %.4f, so the sheet window is %.4f < r < %.4f)"
          % (a_s3, 2.5 * rho_big, 0.8 * a_s3))
    win = (rg > 2.5 * rho_big) & (rg < 0.8 * a_s3)
    if win.sum() >= 3:
        d_fit = np.polyfit(lo[win], lm[win], 1)[0]
        print("\n  slope in the window: %.3f   (sheet 2, 3-D 3)" % d_fit)
    else:
        d_fit = float("nan")
        print("\n  window too narrow to fit (%d points)" % win.sum())
    return dict(tag=tag, alpha=float(al), dim=float(d_fit),
                L=Ls.tolist(), rho=rhos.tolist(), r=rg.tolist(), m=M.tolist(),
                rho_big=rho_big, a_r3=a_r3, a_s3=float(a_s3))


if __name__ == "__main__":
    out = [run("TREFOIL coil", os.path.join(ROOT, "output/Lsweep"), 0.18,
               os.path.join(ROOT, "output/T2_3_q/T2_3_s3.vect")),
           run("FIGURE-EIGHT coil", os.path.join(ROOT, "output/f8Lsweep"), 0.13,
               os.path.join(ROOT, "output/f8sweep/base_final.vect"))]
    json.dump(out, open(os.path.join(ROOT, "notes/local_dimension.json"), "w"),
              indent=1)
    print("\n\nSUMMARY")
    for o in out:
        print("  %-20s  rho ~ L^%+.3f   mass-radius slope %.2f"
              % (o["tag"], o["alpha"], o["dim"]))
    print("  sheet: alpha = -1, slope 2      3-D: alpha = -1/2, slope 3")
