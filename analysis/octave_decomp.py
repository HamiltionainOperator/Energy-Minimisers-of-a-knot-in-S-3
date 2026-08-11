#!/usr/bin/env python3
"""Space-partition numerics: contribution to E_q = E_{S^3}/L^2 from pairs whose
geodesic separation lies in each OCTAVE [pi/2^{j+1}, pi/2^j).

This is the quantity a space-partition proof has to bound below: one octave =
one level of the recursion  E = (off-diagonal at scale d) + (diagonal, recurse).
Sum over octaves = E_q, so the table is self-validating."""
import numpy as np, sys, json
sys.path.insert(0, "/Users/yash/knot-s3")
from s3_project import R3toS3

def load(p):
    t = open(p).read().split(); n = int(t[1]); k = (len(t) - 2) // n
    X = np.array(t[2:2 + k * n], float).reshape(n, k)
    if k == 3: X = R3toS3(X)
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def octaves(X, J=12, nb=600, seed=0, chunk=8):
    d = np.arccos(np.clip(np.einsum('ij,ij->i', X, np.roll(X, -1, 0)), -1, 1))
    s = np.concatenate([[0.], np.cumsum(d)]); L = s[-1]; s = s[:-1]
    w = .5 * (d + np.roll(d, 1))
    rng = np.random.default_rng(seed)
    cen = rng.choice(len(X), size=min(nb, len(X)), replace=False)
    edges = np.pi * 2.0 ** -np.arange(J + 1)          # pi, pi/2, pi/4, ...
    acc = np.zeros(J)
    for i in range(0, len(cen), chunk):
        c = cen[i:i + chunk]
        D = np.arccos(np.clip(X[c] @ X.T, -1, 1))
        ds = np.abs(s[c][:, None] - s[None, :]); ds = np.minimum(ds, L - ds)
        with np.errstate(divide='ignore', invalid='ignore'):
            f = np.where(ds > 1e-12, 1.0 / np.maximum(D, 1e-12) ** 2 - 1.0 / ds ** 2, 0.0)
        f = np.nan_to_num(f, posinf=0, neginf=0)
        for j in range(J):
            m = (D < edges[j]) & (D >= edges[j + 1])
            acc[j] += (m * f * w[None, :]).sum()
    return edges, acc * (L / len(cen)) / L ** 2, L

CURVES = [("output/Lsweep/G2_k0128.vect",  "coil"),
          ("output/Lsweep/G2_k0320.vect",  "coil"),
          ("output/Lsweep/G3_k00640.vect", "coil"),
          ("output/Lsweep/G3_k01280.vect", "coil"),
          ("output/craster_round/rr_M11.s4", "3-D"),
          ("output/craster_round/rr_M15.s4", "3-D")]
res = []
for p, kind in CURVES:
    e, a, L = octaves(load("/Users/yash/knot-s3/" + p))
    res.append((kind, L, e, a))
    print("  %-5s L=%7.1f   sum over octaves = %.4f" % (kind, L, a.sum()))
json.dump([(k, L, e.tolist(), a.tolist()) for k, L, e, a in res],
          open("/Users/yash/knot-s3/notes/octave_decomp.json", "w"), indent=1)

print("\n  CONTRIBUTION PER OCTAVE  (trefoil coil, sheet)\n")
hdr = [r for r in res if r[0] == "coil"]
print("   octave  delta range          " + "".join("L=%-8.0f" % r[1] for r in hdr))
e = hdr[0][2]
for j in range(len(e) - 1):
    row = "   %2d     %.4f-%.4f   " % (j, e[j + 1], e[j])
    for r in hdr:
        row += "  %7.4f " % r[3][j]
    print(row)
print("\n  CONTRIBUTION PER OCTAVE  (rounded raster, 3-D)\n")
hdr2 = [r for r in res if r[0] == "3-D"]
print("   octave  delta range          " + "".join("L=%-8.0f" % r[1] for r in hdr2))
for j in range(len(e) - 1):
    row = "   %2d     %.4f-%.4f   " % (j, e[j + 1], e[j])
    for r in hdr2:
        row += "  %7.4f " % r[3][j]
    print(row)
