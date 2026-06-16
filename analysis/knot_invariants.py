#!/usr/bin/env sage-python
"""
Knot invariants for a minimized .vect curve, via SageMath.

3D curve → (downsample) → planar diagram → oriented Gauss code → Sage Link,
then report determinant, Alexander & Jones polynomials, and signature.
Picks the diagram matching the majority determinant over a few rotations.

    sage -python analysis/knot_invariants.py output/T2_3/T2_3_s3.vect
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
from sage.all import Link

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vect_to_pdcode import read_vect, find_crossings


def rot(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def downsample(p, t):
    return p if len(p) <= t else p[np.linspace(0, len(p), t, endpoint=False).astype(int)]


def link_from(pts):
    cr = find_crossings(pts, proj_axis=2)
    if not cr:
        return None
    ev = []
    for ci, c in enumerate(cr):
        ev.append((c["param_i"], ci, c["over"] == c["edge_i"]))
        ev.append((c["param_j"], ci, c["over"] == c["edge_j"]))
    ev.sort(key=lambda e: e[0])
    gauss = [(ci + 1) if o else -(ci + 1) for (_, ci, o) in ev]
    signs = [int(c["sign"]) for c in cr]
    return Link([[gauss], signs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vect")
    ap.add_argument("--points", type=int, default=140)
    ap.add_argument("--ntries", type=int, default=6)
    a = ap.parse_args()

    pts = downsample(np.asarray(read_vect(a.vect)[0], float), a.points)
    cand = []
    for s in range(a.ntries):
        try:
            L = link_from(pts @ rot(s).T)
            cand.append((1 if L is None else int(L.determinant()), L))
        except Exception:
            continue
    dets = [d for d, _ in cand]
    print(a.vect)
    if not dets:
        print("  -> no valid diagram")
        return
    det = Counter(dets).most_common(1)[0][0]
    L = next((L for d, L in cand if d == det and L is not None), None)
    print(f"  determinant : {det}    (votes {sorted(Counter(dets).items())})")
    if L is None:
        print("  Alexander   : 1  (unknot — no crossings)")
        return
    for name, fn in [("crossings ", lambda: len(L.pd_code())),
                     ("Alexander ", L.alexander_polynomial),
                     ("Jones     ", L.jones_polynomial),
                     ("signature ", L.signature)]:
        try:
            print(f"  {name}: {fn()}")
        except Exception as e:
            print(f"  {name}: (failed: {e})")


if __name__ == "__main__":
    main()
