#!/usr/bin/env python3
"""Twist-knot generator: m half-twists + clasp.

Construction (single closed curve):
  1. two strands twisted m half-turns around the z-axis (the "band"),
  2. a clasp at the top: the closing arc from strand A dives down the central
     axis (always inside the inter-strand gap), exits sideways through the gap
     half a twist below the top, and rises outside to meet strand B,
  3. a plain cap closing the bottom.

Twist knots: m=1 trefoil(det 3), m=2 figure-eight 4_1(det 5), m=3 5_2(det 7),
m=4 6_1(det 9), m=5 7_2(det 11).  All hyperbolic for m>=2.  det = 2m+1
certifies the construction (verify with analysis/knot_check.py).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_composites import resample_uniform, write_vect  # noqa: E402


def _seg(p, q, n):
    p, q = np.asarray(p, float), np.asarray(q, float)
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    return p + t * (q - p)


def twist_knot(m: int, n_points: int = 1000, clasp_sign: int = +1) -> np.ndarray:
    a, H = 1.0, 1.2 * m          # strand radius, band height (1.2 per half-twist)
    h, g = 0.9, 2.2              # clasp rise, clasp exit radius

    theta = lambda z: np.pi * m * (z / H - 1.0)   # A ends at angle 0 (x=+a)

    # 1. strand A, bottom -> top
    z = np.linspace(0.0, H, 600, endpoint=False)
    A = np.stack([a * np.cos(theta(z)), a * np.sin(theta(z)), z], axis=1)

    # 2. clasp arc A(H) -> B(H): up, to the axis, DOWN the axis (always in the
    #    inter-strand gap), out through the gap half a twist below the top,
    #    up outside, over to B's end.
    Hd = H - 0.55 * (H / m)
    thd = theta(Hd) + clasp_sign * np.pi / 2.0    # gap direction at exit height
    exit_pt = [g * np.cos(thd), g * np.sin(thd), Hd]
    top = H + h
    way = [
        ([a, 0, H], [a, 0, top]),
        ([a, 0, top], [0, 0, top]),
        ([0, 0, top], [0, 0, Hd]),               # down the central axis
        ([0, 0, Hd], exit_pt),                    # out through the gap
        (exit_pt, [g * np.cos(thd), g * np.sin(thd), top + 0.5]),
        ([g * np.cos(thd), g * np.sin(thd), top + 0.5], [-a, 0, top + 0.5]),
        ([-a, 0, top + 0.5], [-a, 0, H]),
    ]
    clasp = np.vstack([_seg(p, q, 60) for p, q in way])

    # 3. strand B, top -> bottom (antipodal to A)
    zb = np.linspace(H, 0.0, 600, endpoint=False)
    B = np.stack([-a * np.cos(theta(zb)), -a * np.sin(theta(zb)), zb], axis=1)

    # 4. bottom cap B(0) -> A(0), a semicircle dipping below z=0 in the y=0 plane
    sgn = (-1.0) ** m                             # A(0) = (a*sgn, 0, 0)
    s = np.linspace(0.0, np.pi, 120, endpoint=False)
    cap = np.stack([-a * sgn * np.cos(s), np.zeros_like(s), -0.8 * np.sin(s)], axis=1)

    return resample_uniform(np.vstack([A, clasp, B, cap]), n_points)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Twist knot (m half-twists + clasp)")
    ap.add_argument("m", type=int, help="half-twists: 1=3_1, 2=4_1, 3=5_2, 4=6_1, 5=7_2")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--clasp-sign", type=int, default=1, choices=[1, -1])
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    pts = twist_knot(args.m, args.n, args.clasp_sign)
    write_vect(args.out, pts)
    print(f"Generated twist knot m={args.m} (expect det {2*args.m+1})  "
          f"(N={len(pts)})  →  {args.out}")
