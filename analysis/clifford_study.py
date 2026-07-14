#!/usr/bin/env python3
"""
Does the E_q-minimising torus knot lie on the Clifford torus?  (matched-n study)

For each T(p,q) at a FIXED resolution n it computes, all at the same n so the
energies are directly comparable:
  * E_q^free   : free minimisation of E_q from the (symmetric) Clifford start
                 (--energy quantity --no-normalize), plus a determinant check
                 that the knot type was preserved;
  * E_q^Cliff  : the BEST energy any on-Clifford placement achieves, from
                 `--cliffordscan p q n` (min over the torus radius r);
  * CV         : std/mean of the projection radius of the FREE result onto its
                 covariance eigen-2-planes (analysis/clifford_check) — ~0 means a
                 genuine constant-radius Clifford torus.

Verdict logic (the decisive test is the energy comparison):
  free >= Cliff*0.99 and CV small   -> ON the Clifford torus (free == Clifford)
  free <  Cliff*0.99                -> OFF: the minimiser beats every Clifford
                                       placement, so it genuinely leaves the torus
  free >  Cliff*1.01 and CV large    -> free run UNDER-PERFORMED the Clifford
                                       optimum (stuck); inconclusive, flagged

Usage:
  python3 analysis/clifford_study.py                      # default set
  python3 analysis/clifford_study.py 2,7,1500,4000 2,33,3000,6000
  (each arg is  p,q,n,iters)
"""
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "energy_s3")
PYEXE = sys.executable
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from clifford_check import lift_to_s3, clifford_diagnostic, read_vect_r3
from knot_check import read_vect_pts, robust_determinant

OUT = os.path.join(ROOT, "output", "_cliffstudy")


def torus_genus(p, q):
    return (p - 1) * (q - 1) // 2


def cliffscan_min_eq(p, q, n):
    out = subprocess.run([BIN, "--cliffordscan", str(p), str(q), str(n)],
                         capture_output=True, text=True).stdout
    m = re.search(r"E_q=([0-9.eE+-]+) @ r=([0-9.]+)\s*$", out.strip().splitlines()[-1])
    return (float(m.group(1)), float(m.group(2))) if m else (float("nan"), float("nan"))


def free_min(p, q, n, iters):
    d = os.path.join(OUT, "T%d_%d_n%d" % (p, q, n))
    os.makedirs(d, exist_ok=True)
    subprocess.run([PYEXE, os.path.join(ROOT, "knots", "generate.py"),
                    str(p), str(q), "--n", str(n), "--out", d],
                   check=True, capture_output=True, text=True)
    init = os.path.join(d, "T%d_%d.vect" % (p, q))
    s3 = os.path.join(d, "T%d_%d_s3.vect" % (p, q))
    log = os.path.join(d, "energy_log.csv")
    proc = subprocess.run([BIN, init, log, s3, str(iters), "0.01",
                           "--energy", "quantity", "--no-normalize"],
                          capture_output=True, text=True)
    m = re.search(r"Final\s+E_q energy\s*:\s*([0-9.eE+-]+)", proc.stdout)
    e_free = float(m.group(1)) if m else float("nan")
    det0, _, _ = robust_determinant(np.asarray(read_vect_pts(init)), ntries=9)
    det1, _, _ = robust_determinant(np.asarray(read_vect_pts(s3)), ntries=9)
    d_free = clifford_diagnostic(lift_to_s3(read_vect_r3(s3)))
    return e_free, s3, det0, det1, d_free


DEFAULT = ["2,3,1500,4000", "2,5,1500,4000", "2,7,1500,4000",
           "2,11,2000,5000", "2,17,2500,5000", "2,33,3500,6000",
           "3,4,1500,5000", "3,5,1500,5000", "3,7,2000,5000"]


def main():
    specs = sys.argv[1:] or DEFAULT
    print("Clifford-torus study — QUANTITY energy E_q (matched-n free vs on-Clifford)")
    print("=" * 92)
    print("%-9s %4s %6s %12s %12s %9s %7s %6s  %s"
          % ("knot", "g", "n", "E_q free", "E_q Cliff", "free/Cl", "CV", "det", "verdict"))
    print("-" * 92)
    rows = []
    for spec in specs:
        p, q, n, it = (int(x) for x in spec.split(","))
        e_cl, r_cl = cliffscan_min_eq(p, q, n)
        e_free, s3, det0, det1, dg = free_min(p, q, n, it)
        cv = max(dg["cv1"], dg["cv2"])
        ratio = e_free / e_cl if e_cl == e_cl else float("nan")
        held = det0 == det1
        if not held:
            verdict = "knot CHANGED (det %s->%s) — discard" % (det0, det1)
        elif ratio < 0.99:
            verdict = "OFF Clifford (beats best torus by %.0f%%)" % (100 * (1 - ratio))
        elif cv < 0.05:
            verdict = "ON Clifford  r=%.3f" % max(dg["r1"], dg["r2"])
        elif ratio > 1.01:
            verdict = "free UNDER-converged (>Clifford) — inconclusive"
        else:
            verdict = "~on Clifford (tie), CV=%.2f" % cv
        print("%-9s %4d %6d %12.5f %12.5f %9.3f %7.3f %6s  %s"
              % ("T(%d,%d)" % (p, q), torus_genus(p, q), n, e_free, e_cl,
                 ratio, cv, "%s" % det1 + ("" if held else "!"), verdict))
        rows.append((p, q, n, e_free, e_cl, r_cl, cv, det1, held, verdict))
    print("-" * 92)
    print("free/Cl < 1  =>  the free minimiser beats every Clifford placement (OFF torus).")
    # markdown
    md = os.path.join(OUT, "clifford_study.md")
    os.makedirs(OUT, exist_ok=True)
    with open(md, "w") as f:
        f.write("# E_q Clifford-torus study (matched n)\n\n")
        f.write("| knot | genus | n | E_q free | E_q on-Clifford (r*) | free/Clifford | CV | det | verdict |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for (p, q, n, ef, ec, rc, cv, det, held, v) in rows:
            f.write("| T(%d,%d) | %d | %d | %.5f | %.5f (r=%.2f) | %.3f | %.3f | %s | %s |\n"
                    % (p, q, torus_genus(p, q), n, ef, ec, rc, ef / ec, cv, det, v))
    print("markdown -> %s" % md)


if __name__ == "__main__":
    main()
