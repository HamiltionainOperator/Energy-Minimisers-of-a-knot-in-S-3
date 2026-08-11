#!/usr/bin/env python3
"""
Isotropic 3-D space-filling curve on S³, built in Hopf coordinates.

    x(eta, xi1, xi2) = (cos eta * e^{i xi1},  sin eta * e^{i xi2}),  eta in (0, pi/2)
    metric  ds^2 = d eta^2 + cos^2 eta d xi1^2 + sin^2 eta d xi2^2
    torus at eta has area  2 pi^2 sin(2 eta);   total volume 2 pi^2.

Recipe (the point is ISOTROPY — intra-torus strand spacing matched to the
inter-torus spacing, so the curve is locally 3-dimensional rather than a stack
of sheets):

  1. N nested tori   eta_k = (pi/2) k/(N+1),  inter-torus spacing pi/(2(N+1)).
  2. on torus k wind a (p,q) curve whose length makes
        area / length = inter-torus spacing
     i.e. length_k = 4 pi N sin(2 eta_k); with p ~ sin eta, q ~ cos eta this is
        (p_k, q_k) ~ 2 sqrt(2) N (sin eta_k, cos eta_k),  gcd forced to 1.
  3. bridge consecutive tori radially at xi = (0,0); close the loop with a
     return radius at a generic xi*, routed through the empty shells.

Total length L ~ 8 N^2, so N ~ L^{1/2} and the spacing ~ 1/N ~ L^{-1/2}: the
3-D signature.  Contrast the raster/boustrophedon snake, which fills volume but
lays its length in parallel lanes and is therefore locally a SHEET.

Writes 4-column S³ files for `build/energy_s3 --eval4`.
"""
import os
import sys

import numpy as np

ROOT = "/Users/yash/knot-s3"
OUT = os.path.join(ROOT, "output/hopf")
XI_RET = (0.5137, 1.2731)          # generic return angle: misses every strand


def hopf(eta, xi1, xi2):
    return np.stack([np.cos(eta) * np.cos(xi1), np.cos(eta) * np.sin(xi1),
                     np.sin(eta) * np.cos(xi2), np.sin(eta) * np.sin(xi2)], -1)


def coprime(p, q):
    p, q = max(1, int(p)), max(1, int(q))
    while np.gcd(p, q) != 1:
        q += 1
    return p, q


def build(N, pts_per_spacing=9):
    dη = (np.pi / 2) / (N + 1)
    rho = dη                                     # target isotropic spacing
    step = rho / pts_per_spacing                 # target S³ edge length
    etas = [(np.pi / 2) * k / (N + 1) for k in range(1, N + 1)]
    segs = []

    def arcsample(f, t0, t1, length):
        m = max(2, int(np.ceil(length / step)))
        return f(np.linspace(t0, t1, m, endpoint=False))

    xi = np.array([0.0, 0.0])                    # current angular position
    xi_start = xi.copy()
    for k, eta in enumerate(etas):
        alpha = 2 * np.sqrt(2) * N
        p, q = coprime(round(alpha * np.sin(eta)), round(alpha * np.cos(eta)))
        speed = np.hypot(p * np.cos(eta), q * np.sin(eta))
        # stop ONE strand spacing short of closing: a fully closed (p,q) loop
        # would return to its entry point and the path would touch itself there
        t_end = 2 * np.pi - rho / speed
        segs.append(arcsample(
            lambda t, e=eta, p=p, q=q, x=xi.copy(): hopf(e, x[0] + p * t, x[1] + q * t),
            0.0, t_end, t_end * speed))
        xi = xi + np.array([p, q]) * t_end       # exit angle
        if k + 1 < len(etas):                    # radial bridge, at the exit angle
            e2 = etas[k + 1]
            segs.append(arcsample(lambda t, x=xi.copy(): hopf(t, x[0], x[1]),
                                  eta, e2, e2 - eta))

    X = np.concatenate(segs, 0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, rho


def s3_len(X):
    return float(np.arccos(np.clip(np.einsum("ij,ij->i", X, np.roll(X, -1, 0)),
                                   -1, 1)).sum())


def min_strand(X, delta, ncent=800, chunk=64, seed=0):
    d = np.arccos(np.clip(np.einsum("ij,ij->i", X, np.roll(X, -1, 0)), -1, 1))
    s = np.concatenate([[0.0], np.cumsum(d)]); L = s[-1]; s = s[:-1]
    rng = np.random.default_rng(seed)
    cen = rng.choice(len(X), size=min(ncent, len(X)), replace=False)
    nn = np.empty(len(cen))
    for i in range(0, len(cen), chunk):
        c = cen[i:i + chunk]
        D = np.arccos(np.clip(X[c] @ X.T, -1, 1))
        ds = np.abs(s[c][:, None] - s[None, :]); ds = np.minimum(ds, L - ds)
        D[ds < delta] = np.inf
        nn[i:i + chunk] = D.min(1)
    nn = nn[np.isfinite(nn)]
    return float(nn.mean()), float(nn.min())


def write4(X, path):
    with open(path, "w") as f:
        f.write("1\n%d\n" % len(X))
        for r in X:
            f.write("%.17g %.17g %.17g %.17g\n" % tuple(r))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    Ns = [int(x) for x in (sys.argv[1:] or [4, 6, 8, 10, 12, 14])]
    print("   N       n        L      target rho   mean nn   min nn   L/(8N^2)")
    for N in Ns:
        X, rho = build(N)
        L = s3_len(X)
        mu, mn = min_strand(X, delta=3 * rho)
        p = os.path.join(OUT, "hopf_N%02d.s4" % N)
        write4(X, p)
        print("  %2d  %7d  %8.2f    %.5f    %.5f  %.5f    %.3f"
              % (N, len(X), L, rho, mu, mn, L / (8.0 * N * N)))
