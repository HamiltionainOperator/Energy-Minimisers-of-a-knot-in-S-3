#!/usr/bin/env python3
"""4-strand plat-closure knot generator (2-bridge / rational knots).

A braid word on 4 strands (generators 1,2,3; sign = crossing sign) is drawn as
strand paths in 3D, then plat-closed: caps join columns (0,1) and (2,3) at both
ends.  Twist knots are the 2-bridge knots b(2m+1,2) = plat closure of
sigma_2^m sigma_1^2:

    m=1 -> 3_1 (det 3)    m=2 -> 4_1 (det 5)    m=3 -> 5_2 (det 7)
    m=4 -> 6_1 (det 9)    m=5 -> 7_2 (det 11)

Verify with analysis/knot_check.py (det = 2m+1 certifies the construction).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_composites import resample_uniform, write_vect  # noqa: E402

XCOL = [0.0, 1.0, 2.0, 3.0]     # strand column x-positions
DZ = 1.5                         # height per braid letter
BUMP = 0.85                      # y-excursion of the over/under strands
SAMP = 40                        # samples per letter per strand


def braid_strands(word):
    """Trace 4 strand paths through the braid. Returns (paths, perm) where
    paths[k] is the (x,y,z) polyline of the strand STARTING at column k and
    perm[k] is the column it ENDS at."""
    pos = list(range(4))                    # pos[k] = current column of strand k
    paths = [[(XCOL[k], 0.0, 0.0)] for k in range(4)]
    z = 0.0
    for step, g in enumerate(word):
        i = abs(g) - 1                      # acts on columns i, i+1
        t = np.linspace(0.0, 1.0, SAMP, endpoint=False)[1:]
        for k in range(4):
            c = pos[k]
            if c == i or c == i + 1:
                c2 = i + 1 if c == i else i
                x = XCOL[c] + (XCOL[c2] - XCOL[c]) * (1 - np.cos(np.pi * t)) / 2
                # crossing sign decides which strand bulges over (+y) vs under
                over = (c == i) == (g > 0)
                y = (BUMP if over else -BUMP) * np.sin(np.pi * t)
                for xx, yy, tt in zip(x, y, t):
                    paths[k].append((xx, yy, z + DZ * tt))
            else:
                for tt in t:
                    paths[k].append((XCOL[c], 0.0, z + DZ * tt))
        # update positions
        a = [k for k in range(4) if pos[k] == i][0]
        b = [k for k in range(4) if pos[k] == i + 1][0]
        pos[a], pos[b] = i + 1, i
        z += DZ
        for k in range(4):
            paths[k].append((XCOL[pos[k]], 0.0, z))
    return [np.array(p) for p in paths], pos


def cap(c1, c2, z, direction):
    """Semicircular cap joining columns c1 -> c2 at height z, bulging in z."""
    s = np.linspace(0.0, np.pi, 40, endpoint=False)[1:]
    x = XCOL[c1] + (XCOL[c2] - XCOL[c1]) * (1 - np.cos(s)) / 2
    zz = z + direction * 0.8 * np.sin(s)
    return np.stack([x, np.zeros_like(x), zz], axis=1)


def plat_closure(word, n_points=1000):
    paths, endpos = braid_strands(word)
    top_z = DZ * len(word)
    start_col = {k: paths[k][0][0] for k in range(4)}
    partner = {0: 1, 1: 0, 2: 3, 3: 2}
    # index strands by their start column / end column
    by_start = {int(round(paths[k][0][0])): k for k in range(4)}
    by_end = {endpos[k]: k for k in range(4)}

    pts, used = [], set()
    col, going_up = 0, True                  # start at bottom of column 0
    while True:
        if going_up:
            k = by_start[col]
            if k in used:
                break
            used.add(k)
            pts.append(paths[k])
            end_col = endpos[k]
            pts.append(cap(end_col, partner[end_col], top_z, +1))   # top cap
            col, going_up = partner[end_col], False
        else:
            k = by_end[col]
            if k in used:
                break
            used.add(k)
            pts.append(paths[k][::-1])
            start_c = int(round(paths[k][0][0]))
            pts.append(cap(start_c, partner[start_c], 0.0, -1))     # bottom cap
            col, going_up = partner[start_c], True
    if len(used) != 4:
        raise ValueError("plat closure has more than one component for this word")
    return resample_uniform(np.vstack(pts), n_points)


# twist knot = b(2m+1,2), odd continued fraction [m,1,1] -> sigma_2^m sigma_1^-1 sigma_2
# (a sigma_1 block adjacent to a cap is a removable curl, so the middle block
# must be sandwiched between sigma_2 blocks; validated m=1..5 -> det 3,5,7,9,11)
TWIST_WORDS = {m: [2] * m + [-1, 2] for m in range(1, 64)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="4-plat knot generator")
    ap.add_argument("--twist", type=int, help="twist knot: m half-twists (det 2m+1)")
    ap.add_argument("--word", type=int, nargs="+",
                    help="explicit braid word, e.g. --word 2 2 2 1 1")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    word = TWIST_WORDS[args.twist] if args.twist else args.word
    pts = plat_closure(word, args.n)
    write_vect(args.out, pts)
    tag = f"twist m={args.twist} (expect det {2*args.twist+1})" if args.twist \
        else f"word {word}"
    print(f"Generated 4-plat {tag}  (N={len(pts)})  →  {args.out}")
