#!/usr/bin/env python3
"""
UNKNOTTED 3-D space-filling curve on S³ (cubulated boustrophedon raster).

d([-1,1]^4) is eight 3-cubes; radially project to S³.  Inside each 3-cube lay a
boustrophedon raster on the lattice hZ^3 (serpentine within a layer, step to the
next layer, serpentine back); join the eight cubes in a cycle across shared
2-faces so the whole thing is one closed curve.

Unknottedness is by construction: each planar serpentine layer retracts to a
straight segment inside its own plane, sweeping through no other layer; doing
that layer by layer leaves a staircase, which unknots to a circle.

The eight cubes are chained along the Hamiltonian cycle
    (0,+) (1,+) (2,+) (3,+) (0,-) (1,-) (2,-) (3,-)
of the facet adjacency graph.  For the chain to close up with NO extra joining
arcs, each cube must be entered and left at OPPOSITE corners of its own cube.
Solving that constraint around the cycle has a unique consistent solution (the
P_k below); the assert verifies it.

Length scales as L ~ h^-2, so sweeping h sweeps L.
Writes 4-column S³ files for `build/energy_s3 --eval4`.
"""
import os
import sys

import numpy as np

ROOT = "/Users/yash/knot-s3"
OUT = os.path.join(ROOT, "output/craster")

# facet (axis, sign) in Hamiltonian-cycle order
FACETS = [(0, 1), (1, 1), (2, 1), (3, 1), (0, -1), (1, -1), (2, -1), (3, -1)]

# P[k] = the 4-cube corner where facet k hands over to facet k+1
P = [(+1, +1, -1, +1), (-1, +1, +1, -1), (+1, -1, +1, +1), (-1, +1, -1, +1),
     (-1, -1, +1, -1), (+1, -1, -1, +1), (-1, +1, -1, -1), (+1, -1, +1, -1)]


def local_axes(ax):
    return [a for a in range(4) if a != ax]


def to_local(facet, p4):
    return np.array([p4[a] for a in local_axes(facet[0])], float)


def to_4d(facet, loc):
    ax, s = facet
    out = np.empty(loc.shape[:-1] + (4,))
    out[..., ax] = s
    for j, a in enumerate(local_axes(ax)):
        out[..., a] = loc[..., j]
    return out


def check_cycle():
    for k, f in enumerate(FACETS):
        entry = to_local(f, P[k - 1])
        exit_ = to_local(f, P[k])
        assert np.allclose(exit_, -entry), (k, entry, exit_)
        assert P[k][f[0]] == f[1] and P[k][FACETS[(k + 1) % 8][0]] == FACETS[(k + 1) % 8][1]
    return True


def raster_corners(M, sigma, aniso=1):
    """serpentine corner vertices in local coords; enters near -sigma, exits near
    +sigma.  aniso>1 stretches the LAYER spacing to aniso*h while keeping the
    in-layer spacing at h, so the curve fills the same region as a stack of
    separated sheets instead of an isotropic 3-D array."""
    h = 2.0 / M
    t = -1.0 + h / 2 + np.arange(M) * h
    K = max(1, M // aniso)                   # fewer, more widely spaced layers
    H = 2.0 / K
    tk = -1.0 + H / 2 + np.arange(K) * H
    V = []
    for k in range(K):                       # layers
        rows = range(M) if k % 2 == 0 else range(M - 1, -1, -1)
        for j in rows:                       # rows within the layer
            fwd = (j + k) % 2 == 0
            a, b = (t[0], t[-1]) if fwd else (t[-1], t[0])
            V.append((a, t[j], tk[k]))
            V.append((b, t[j], tk[k]))
    V = np.array(V) * np.asarray(sigma, float)[None, :]
    if len(V) % 2:                            # keep entry/exit at opposite corners
        V = V[:-1]
    return V, h


def cut_trefoil(a, b, delta=0.15):
    """A trefoil tied into the segment a->b: the standard closed trefoil cut open
    at t=+-delta, then mapped by a similarity so its two ends land on a and b.
    Closing the ends by the chord gives back a trefoil, so splicing this into a
    curve connect-sums a trefoil into it.  Diameter is ~3.1 |b-a|, so with
    |b-a| = h/8 the tangle stays inside a ball of radius ~0.2h and cannot reach
    the neighbouring strands at distance h."""
    t = np.linspace(delta, 2 * np.pi - delta, 400)
    G = np.stack([(2 + np.cos(3 * t)) * np.cos(2 * t),
                  (2 + np.cos(3 * t)) * np.sin(2 * t), np.sin(3 * t)], -1)
    p, q = G[0], G[-1]
    scale = np.linalg.norm(b - a) / np.linalg.norm(q - p)
    # rotation taking (q-p)/|q-p| to (b-a)/|b-a|  (Rodrigues)
    u = (q - p) / np.linalg.norm(q - p); v = (b - a) / np.linalg.norm(b - a)
    w = np.cross(u, v); c = u @ v
    if np.linalg.norm(w) < 1e-12:
        R = np.eye(3) * np.sign(c)
    else:
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        R = np.eye(3) + K + K @ K / (1 + c)
    return a + ((G - p) * scale) @ R.T


def round_corners(V, r, nb=14):
    """Replace every corner of a closed polyline by a quadratic Bezier fillet:
    cut back t = r tan(phi/2) along both incident segments and interpolate with
    the corner as control point.  The result is C^1 with bounded curvature
    (C^{1,1}), which is what the O'Hara energy needs to be FINITE — a polyline
    corner has logarithmically divergent energy, so the un-rounded raster is not
    an admissible curve at all."""
    n = len(V)
    out = []
    for i in range(n):
        p, a, b = V[i], V[(i - 1) % n], V[(i + 1) % n]
        du, dv = p - a, b - p
        lu, lv = np.linalg.norm(du), np.linalg.norm(dv)
        if lu < 1e-14 or lv < 1e-14:
            continue
        u, v = du / lu, dv / lv
        c = np.clip(u @ v, -1, 1)
        phi = np.arccos(c)
        if phi < 1e-9:                     # straight through: keep the vertex
            out.append(p[None, :]); continue
        t = min(r * np.tan(phi / 2), 0.45 * lu, 0.45 * lv)
        A, B = p - t * u, p + t * v
        s_ = np.linspace(0, 1, nb)[:, None]
        out.append((1 - s_) ** 2 * A + 2 * (1 - s_) * s_ * p + s_ ** 2 * B)
    return np.concatenate(out, 0)


def refine(V4, step):
    """subdivide a 4-D polyline so no edge exceeds `step` (open, no wrap)"""
    out = []
    for a, b in zip(V4[:-1], V4[1:]):
        d = np.linalg.norm(b - a)
        m = max(1, int(np.ceil(d / step)))
        out.append(a[None, :] + np.linspace(0, 1, m, endpoint=False)[:, None] * (b - a))
    return np.concatenate(out, 0)


def build(M, pts_per_h=8, tie_trefoil=False, aniso=1, round_r=None):
    assert M % 2 == 1, "M must be odd so the raster runs corner to corner"
    check_cycle()
    h = 2.0 / M
    step = h / pts_per_h
    chain = []
    for k, f in enumerate(FACETS):
        c_in = to_local(f, P[k - 1])
        sigma = -c_in                          # entry near c_in, exit near -c_in
        V, _ = raster_corners(M, sigma, aniso)
        if tie_trefoil and k == 0:
            # splice a small trefoil into the middle of a row of the first cube
            i = 2 * (M * M // 2)                     # start of a mid-cube row
            a = V[i] + 0.5 * (V[i + 1] - V[i])
            b = a + (V[i + 1] - V[i]) / np.linalg.norm(V[i + 1] - V[i]) * (h / 8)
            V = np.vstack([V[:i + 1], a[None, :], cut_trefoil(a, b), b[None, :],
                           V[i + 1:]])
        chain.append(to_4d(f, V))
    V4 = np.concatenate(chain, 0)
    if round_r:                                # C^{1,1} fillets, radius round_r*h
        # nb must scale with the sampling, else the Bezier polyline's own corners
        # become visible at high resolution
        V4 = round_corners(V4, round_r * h, nb=max(14, 3 * pts_per_h))
    V4 = np.vstack([V4, V4[:1]])               # close the loop
    X = refine(V4, step)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)   # radial projection to S³
    return X, h


def s3_len(X):
    return float(np.arccos(np.clip(np.einsum("ij,ij->i", X, np.roll(X, -1, 0)),
                                   -1, 1)).sum())


def nn_strand(X, delta, ncent=900, chunk=64, seed=0):
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


def write_vect(X, path):
    """stereographic projection to R³ for the topology oracle; the pole is put
    at the point of S³ furthest from the curve."""
    k = np.argmax(np.arccos(np.clip(X @ X.T, -1, 1)).min(1)) if len(X) < 4000 else 0
    # rotate a far-from-curve direction to the north pole
    cand = np.eye(4)
    best, R = -1, np.eye(4)
    for c in np.vstack([cand, -cand, np.ones((1, 4)) / 2, -np.ones((1, 4)) / 2]):
        d = np.arccos(np.clip(X @ c, -1, 1)).min()
        if d > best:
            best, pole = d, c
    e4 = np.zeros(4); e4[3] = 1.0
    v = e4 - pole
    R = np.eye(4) - 2 * np.outer(v, v) / (v @ v) if v @ v > 1e-12 else np.eye(4)
    Y = X @ R.T
    with open(path, "w") as f:
        f.write("1\n%d\n" % len(Y))
        for r in Y:
            f.write("%.12g %.12g %.12g\n" % tuple(r[:3] / (1 - r[3])))
    return best


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    Ms = [int(x) for x in (sys.argv[1:] or [3, 5, 7, 9, 11, 13, 15])]
    print("   M     h        n         L      mean nn    min nn   nn/h")
    for M in Ms:
        X, h = build(M)
        L = s3_len(X)
        mu, mn = nn_strand(X, delta=3 * h)
        write4(X, os.path.join(OUT, "craster_M%02d.s4" % M))
        print("  %2d  %.4f  %7d  %8.2f   %.5f  %.5f   %.2f"
              % (M, h, len(X), L, mu, mn, mu / h))
