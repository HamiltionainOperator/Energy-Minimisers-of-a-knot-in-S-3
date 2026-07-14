#!/usr/bin/env python3
"""
Release-from-Clifford test: is the OPTIMAL Clifford-torus placement of T(p,q)
stable under the E_q flow, or does the knot leave the torus?

This is the cleanest, fastest test for large q (where free minimisation from the
symmetric start is slow and risks losing the knot type):

  1. build T(p,q) ON the Clifford torus T_{r*}, where r* is the radius that
     MINIMISES E_q on the torus (from --cliffordscan).  This is the best the
     knot can do without leaving the torus — and it is a valid embedded knot, so
     topology is correct by construction.
  2. minimise E_q from there (--energy quantity --no-normalize).
  3. if the energy DROPS below the on-Clifford value and the projection radius
     stops being constant (CV grows), the Clifford torus is NOT a minimiser:
     the knot has LEFT it.  If it stays (energy flat, CV ~ 0), it is on-Clifford.

Each row: p,q,r*,n,iters.  r* comes from `--cliffordscan p q n` (MIN E_q @ r).

Usage:
  python3 analysis/clifford_release.py 2,11 2,17 2,33 3,7 3,37
  python3 analysis/clifford_release.py --n 2500 --iters 3000 2,33
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "energy_s3")
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from clifford_check import lift_to_s3, clifford_diagnostic, read_vect_r3
from knot_check import read_vect_pts, robust_determinant

OUT = os.path.join(ROOT, "output", "_release")


def clifford_pq_s3(p, q, r, n):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    s = np.sqrt(max(0.0, 1.0 - r * r))
    return np.column_stack([r * np.cos(p * th), r * np.sin(p * th),
                            s * np.cos(q * th), s * np.sin(q * th)])


def s3_to_r3(x):
    """Stereographic S^3 -> R^3, matching the C++ s3_to_r3 (y = x[:3]/(1-x3))."""
    denom = (1.0 - x[:, 3])[:, None]
    return x[:, :3] / denom


def write_vect(pts_r3, path):
    with open(path, "w") as f:
        f.write("1\n%d\n" % len(pts_r3))
        for x, y, z in pts_r3:
            f.write("%.10f %.10f %.10f\n" % (x, y, z))


def cliffscan_min(p, q, n):
    out = subprocess.run([BIN, "--cliffordscan", str(p), str(q), str(n)],
                         capture_output=True, text=True).stdout
    m = re.search(r"E_q=([0-9.eE+-]+) @ r=([0-9.]+)\s*$", out.strip().splitlines()[-1])
    return (float(m.group(1)), float(m.group(2))) if m else (float("nan"), float("nan"))


def run(p, q, n, iters):
    e_cl, r_star = cliffscan_min(p, q, n)
    d = os.path.join(OUT, "T%d_%d" % (p, q))
    os.makedirs(d, exist_ok=True)
    start = os.path.join(d, "start.vect")
    s3 = os.path.join(d, "T%d_%d_s3.vect" % (p, q))
    log = os.path.join(d, "energy_log.csv")
    # build the optimal on-Clifford knot as the starting curve
    x = clifford_pq_s3(p, q, r_star, n)
    write_vect(s3_to_r3(x), start)
    proc = subprocess.run([BIN, start, log, s3, str(iters), "0.01",
                           "--energy", "quantity", "--no-normalize"],
                          capture_output=True, text=True)
    e0 = float(re.search(r"Initial E_q energy:\s*([0-9.eE+-]+)", proc.stdout).group(1))
    ef = float(re.search(r"Final\s+E_q energy\s*:\s*([0-9.eE+-]+)", proc.stdout).group(1))
    dg0 = clifford_diagnostic(lift_to_s3(read_vect_r3(start)))
    dg1 = clifford_diagnostic(lift_to_s3(read_vect_r3(s3)))
    det0, _, _ = robust_determinant(np.asarray(read_vect_pts(start)), ntries=7)
    det1, _, _ = robust_determinant(np.asarray(read_vect_pts(s3)), ntries=7)
    return dict(p=p, q=q, n=n, r_star=r_star, e_cl=e_cl, e0=e0, ef=ef,
                cv0=max(dg0["cv1"], dg0["cv2"]), cv1=max(dg1["cv1"], dg1["cv2"]),
                det0=det0, det1=det1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("knots", nargs="+", help="p,q specs e.g. 2,17 2,33 3,7")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--iters", type=int, default=3000)
    args = ap.parse_args()

    print("Release-from-Clifford test — E_q  (start ON optimal Clifford torus)")
    print("=" * 96)
    print("%-9s %4s %7s %11s %11s %9s %8s %8s  %s"
          % ("knot", "n", "r*", "E_q start", "E_q final", "drop", "CV start",
             "CV final", "verdict"))
    print("-" * 96)
    for spec in args.knots:
        p, q = (int(v) for v in spec.split(","))
        try:
            r = run(p, q, args.n, args.iters)
        except Exception as exc:
            print("%-9s  ERROR: %s" % ("T(%d,%d)" % (p, q), exc))
            continue
        drop = 1.0 - r["ef"] / r["e0"]
        held = r["det0"] == r["det1"]
        if not held:
            v = "knot CHANGED (%s->%s)" % (r["det0"], r["det1"])
        elif drop > 0.01 and r["cv1"] > 0.08:
            v = "LEFT the torus (off-Clifford)"
        elif drop <= 0.01 and r["cv1"] < 0.05:
            v = "STAYED on the torus"
        else:
            v = "partial (drop %.1f%%, CV %.2f)" % (100 * drop, r["cv1"])
        print("%-9s %4d %7.3f %11.5f %11.5f %8.1f%% %8.3f %8.3f  %s"
              % ("T(%d,%d)" % (p, q), r["n"], r["r_star"], r["e0"], r["ef"],
                 100 * drop, r["cv0"], r["cv1"], v))
    print("-" * 96)
    print("Start sits ON the best Clifford torus (CV start ~ 0, E_q start = on-Clifford min).")
    print("A drop in E_q with CV rising => the Clifford torus is unstable => minimiser is OFF it.")


if __name__ == "__main__":
    main()
