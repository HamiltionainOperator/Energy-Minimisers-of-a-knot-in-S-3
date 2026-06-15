#!/usr/bin/env python3
"""
Robust knot-type verification for .vect curves.

Why this exists
---------------
`pyknotid.Knot(pts).determinant()` projects the curve to a single fixed plane
to read off a Gauss code.  For some curves that default projection is
*degenerate* — a crossing lines up exactly behind another and is missed — so a
genuine trefoil can report only 2 crossings and a determinant of 1 (unknot).
This happened for the freshly generated T(2,3) torus knot.

The determinant is a topological invariant, so *generic* projections all give
the correct value; only a measure-zero set of viewing directions is degenerate.
We therefore evaluate it under several random 3-D rotations and take the
majority vote.  A unanimous vote is strong evidence; a split vote flags a curve
that is geometrically near-degenerate (e.g. a strand pair on the verge of a
tunnelling event) and deserves a closer look.

Usage
-----
    python3 analysis/knot_check.py output/Granny/Granny_s3.vect --expected 9
    python3 analysis/knot_check.py output/T2_3/T2_3_s3.vect --expected 3 --ntries 15

Exit code: 0 if the (robust) determinant matches --expected (or none given), 1 otherwise.
"""
import argparse
import sys
import warnings
from collections import Counter

import numpy as np

# pyknotid still references a few numpy aliases removed in newer numpy.
np.float = float          # type: ignore[attr-defined]
np.int = int              # type: ignore[attr-defined]
np.complex = complex      # type: ignore[attr-defined]
np.long = np.int64        # type: ignore[attr-defined]
warnings.filterwarnings("ignore")


def read_vect_pts(path: str) -> np.ndarray:
    """Read component 0 of a .vect file as an (N,3) array."""
    with open(path) as f:
        f.readline()                       # n_components
        n = int(f.readline())
        return np.array([[float(v) for v in f.readline().split()]
                         for _ in range(n)])


def _random_rotation(seed: int) -> np.ndarray:
    """A uniformly-distributed proper rotation matrix (det +1)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))               # fix QR sign ambiguity
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1                      # ensure proper rotation
    return q


def robust_determinant(pts: np.ndarray, ntries: int = 11):
    """
    Knot determinant by majority vote over `ntries` random rotations.

    Returns (determinant, distribution, unanimous) where
      determinant  : int  — the modal value (None if every attempt failed)
      distribution : list of (value, count) tuples, sorted
      unanimous    : bool — True iff a single value won every successful vote
    """
    from pyknotid.spacecurves import Knot
    votes = []
    for seed in range(ntries):
        rot = _random_rotation(seed)
        try:
            k = Knot(pts @ rot.T, verbose=False)
            votes.append(int(abs(k.determinant())))
        except Exception:
            continue
    if not votes:
        return None, [], False
    counts = Counter(votes)
    det, top = counts.most_common(1)[0]
    return det, sorted(counts.items()), (len(counts) == 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Robust knot-type check for a .vect file")
    ap.add_argument("vect", help="input .vect file")
    ap.add_argument("--expected", type=int, default=None,
                    help="expected determinant (e.g. trefoil=3, granny=9)")
    ap.add_argument("--ntries", type=int, default=11,
                    help="number of random-rotation votes (default 11)")
    args = ap.parse_args()

    pts = read_vect_pts(args.vect)
    det, dist, unanimous = robust_determinant(pts, args.ntries)

    print(f"{args.vect}")
    print(f"  robust determinant : {det}")
    print(f"  vote distribution  : {dist}  ({'unanimous' if unanimous else 'SPLIT — near-degenerate geometry'})")

    if args.expected is None:
        sys.exit(0)
    ok = (det == args.expected)
    print(f"  expected           : {args.expected}  -> {'PASS' if ok else 'FAIL'}")
    if not unanimous:
        print("  WARNING: split vote — the curve may be close to a self-crossing; "
              "inspect before trusting the topology.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
