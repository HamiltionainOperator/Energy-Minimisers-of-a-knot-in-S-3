#!/usr/bin/env sage-python
"""
Robust knot-type verification under SageMath.

3D curve (.vect)  →  random rotation  →  planar diagram  →  oriented Gauss code
                  →  Sage's Link.determinant()  →  majority vote.

Why Sage: Sage's Link simplifies the diagram before computing invariants, which is
far more reliable than a raw point-cloud determinant at high vertex counts. We also
downsample to a few hundred points first, so finely-spaced near-tangent crossings
(the source of the split votes at N=5000) collapse to clean, well-separated ones.

Run it through Sage's Python:

    sage -python analysis/knot_check_sage.py output/T2_5/T2_5_s3.vect --expected 5

A connect sum's determinant is the product of its components' (T(2,3)#T(2,5) → 15).
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
from sage.all import Link

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vect_to_pdcode import read_vect, find_crossings


def random_rotation(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def downsample(pts, target):
    n = len(pts)
    if n <= target:
        return pts
    idx = np.linspace(0, n, target, endpoint=False).astype(int)
    return pts[idx]


def gauss_determinant(pts):
    """Determinant of the knot from one planar projection via Sage's Link."""
    cr = find_crossings(pts, proj_axis=2)
    if not cr:
        return 1  # no crossings in this projection → unknot
    events = []
    for ci, c in enumerate(cr):
        events.append((c["param_i"], ci, c["over"] == c["edge_i"]))
        events.append((c["param_j"], ci, c["over"] == c["edge_j"]))
    events.sort(key=lambda e: e[0])
    gauss = [(ci + 1) if over else -(ci + 1) for (_, ci, over) in events]
    signs = [int(c["sign"]) for c in cr]
    return int(Link([[gauss], signs]).determinant())


def robust_determinant(pts, ntries=11, target=300):
    pts = downsample(np.asarray(pts, dtype=float), target)
    votes = []
    for s in range(ntries):
        try:
            votes.append(gauss_determinant(pts @ random_rotation(s).T))
        except Exception:
            continue
        # early exit: 5 unanimous votes is a confident, clean answer
        if len(votes) >= 5 and len(set(votes)) == 1:
            break
    if not votes:
        return None, [], False
    counts = Counter(votes)
    det, _ = counts.most_common(1)[0]
    return det, sorted(counts.items()), (len(counts) == 1)


def main():
    ap = argparse.ArgumentParser(description="Robust knot determinant via SageMath")
    ap.add_argument("vect")
    ap.add_argument("--expected", type=int, default=None)
    ap.add_argument("--ntries", type=int, default=11)
    ap.add_argument("--points", type=int, default=300,
                    help="downsample target before crossing detection (default 300)")
    a = ap.parse_args()

    pts = read_vect(a.vect)[0]
    det, dist, unanimous = robust_determinant(pts, a.ntries, a.points)

    print(a.vect)
    print(f"  robust determinant : {det}")
    print(f"  vote distribution  : {dist}  "
          f"({'unanimous' if unanimous else 'SPLIT — near-degenerate geometry'})")
    if a.expected is None:
        sys.exit(0)
    ok = (det == a.expected)
    print(f"  expected           : {a.expected}  -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
