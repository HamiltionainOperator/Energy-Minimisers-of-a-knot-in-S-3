#!/usr/bin/env python3
"""
Does a minimised knot lie on a Clifford torus in S^3?

A Clifford torus  T_r = { (z,w) in S^3 ⊂ C^2 : |z| = r, |w| = sqrt(1-r^2) }  is
the product of two circles living in two ORTHOGONAL 2-planes of R^4.  A curve
lies on SOME Clifford torus iff R^4 splits into orthogonal 2-planes P1 ⊥ P2 such
that every point's projection onto P1 has constant radius r (and onto P2,
sqrt(1-r^2)).

Rotation-invariant diagnostic (we don't need to know the planes in advance):
  1. lift the .vect (R^3 stereographic image) back to S^3 with the inverse
     stereographic map, matching the C++ r3_to_s3 EXACTLY:
        r2=|y|^2, d=1/(r2+1), x=(2y0 d, 2y1 d, 2y2 d, (r2-1) d).
  2. covariance  M = (1/n) sum_i x_i x_i^T   (4x4 symmetric, trace = 1).
  3. on a Clifford torus the eigenvalues PAIR as (a,a,b,b): within each plane the
     constant-radius circle puts r^2/2 on each of its two axes.  The eigen-2-
     planes P1=span(v1,v2), P2=span(v3,v4) are then the Clifford planes.
  4. project every point onto P1: rho_i = sqrt(<x,v1>^2 + <x,v2>^2).  On a
     Clifford torus rho_i is CONSTANT = r  =>  CV = std(rho)/mean(rho) -> 0.

Verdict: ON a Clifford torus iff the eigenvalues pair AND rho is near-constant
(CV below --cv-thresh, default 0.05).  Inferred radius r = mean(rho) on the
larger-variance plane;  the two radii satisfy r1^2 + r2^2 = 1.

CAVEAT — the symmetric case r = 1/sqrt(2): then |z|=|w| and M is isotropic
(all eigenvalues 1/4), so the eigen-planes are UNDETERMINED from M alone.  We
detect this (tiny between-pair gap) and flag it: a near-isotropic M means "can't
tell from covariance" rather than "off the torus."  The quantity-energy trefoil
minimiser sits at r~0.776 (clearly asymmetric), so this is decisive there.

Usage:
  python3 analysis/clifford_check.py output/granny_q/granny_s3.vect
  python3 analysis/clifford_check.py output/_qscan/T2_3_q/T2_3_s3.vect a.vect b.vect
  python3 analysis/clifford_check.py --selftest        # analytic controls
"""
import argparse
import os
import sys

import numpy as np


# ── .vect reader (reuse the validated pipeline parser; fall back to inline) ────
def read_vect_r3(path):
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from knot_check import read_vect_pts
        return np.asarray(read_vect_pts(path), float)
    except Exception:
        pts = []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        pts.append([float(v) for v in parts])
                    except ValueError:
                        pass
        return np.asarray(pts, float)


def lift_to_s3(y):
    """Inverse stereographic R^3 -> S^3, identical to the C++ r3_to_s3."""
    r2 = (y * y).sum(1)
    d = 1.0 / (r2 + 1.0)
    x = np.empty((len(y), 4))
    x[:, 0] = 2.0 * y[:, 0] * d
    x[:, 1] = 2.0 * y[:, 1] * d
    x[:, 2] = 2.0 * y[:, 2] * d
    x[:, 3] = (r2 - 1.0) * d
    return x


def clifford_diagnostic(x, cv_thresh=0.05, degen_gap=0.02):
    """x: (n,4) points on S^3. Returns a dict of diagnostics + verdict."""
    n = len(x)
    M = (x.T @ x) / n                      # 4x4 covariance, trace ~ 1
    w, V = np.linalg.eigh(M)               # ascending
    w = w[::-1]
    V = V[:, ::-1]                         # columns = eigvecs, descending eigval

    # pair (w0,w1) and (w2,w3); P1 = bigger-radius plane
    P1 = V[:, [0, 1]]
    P2 = V[:, [2, 3]]
    c1 = x @ P1                            # (n,2) coords in plane 1
    c2 = x @ P2
    rho1 = np.linalg.norm(c1, axis=1)
    rho2 = np.linalg.norm(c2, axis=1)
    cv1 = rho1.std() / rho1.mean() if rho1.mean() > 1e-9 else 0.0
    cv2 = rho2.std() / rho2.mean() if rho2.mean() > 1e-9 else 0.0
    # one circle collapsed (r->0) => the curve is a great circle in a single
    # 2-plane, a DEGENERATE Clifford torus, not a genuine 2-torus.
    degenerate = min(rho1.mean(), rho2.mean()) < 1e-3

    within = (w[0] - w[1]) + (w[2] - w[3])  # gaps INSIDE the pairs (want ~0)
    between = w[1] - w[2]                    # gap BETWEEN the pairs
    spread = w[0] - w[3] + 1e-15
    pair_defect = within / spread           # small => eigenvalues pair cleanly

    isotropic = between < degen_gap and within < degen_gap
    paired = pair_defect < 0.25             # within-pair gaps << overall spread
    constant_radius = max(cv1, cv2) < cv_thresh
    on_clifford = (paired or isotropic) and constant_radius

    return {
        "n": n, "eigs": w, "P1": P1, "P2": P2,
        "r1": rho1.mean(), "r2": rho2.mean(), "cv1": cv1, "cv2": cv2,
        "pair_defect": pair_defect, "between": between, "within": within,
        "isotropic": isotropic, "paired": paired, "degenerate": degenerate,
        "constant_radius": constant_radius, "on_clifford": on_clifford,
    }


def report(tag, d):
    w = d["eigs"]
    print("  %s" % tag)
    print("    eigenvalues (desc) : [%.4f %.4f %.4f %.4f]   sum=%.4f"
          % (w[0], w[1], w[2], w[3], w.sum()))
    print("    pair defect        : %.4f   (within=%.4f, between-pair gap=%.4f)"
          % (d["pair_defect"], d["within"], d["between"]))
    print("    proj-radius r1,r2  : %.4f , %.4f   (r1^2+r2^2=%.4f)"
          % (d["r1"], d["r2"], d["r1"] ** 2 + d["r2"] ** 2))
    print("    radius CV (std/mean): %.4f , %.4f   %s"
          % (d["cv1"], d["cv2"], "(constant ✓)" if d["constant_radius"]
             else "(NOT constant)"))
    if d["isotropic"]:
        print("    NOTE: covariance ~isotropic (near symmetric r=1/√2); "
              "planes undetermined from M.")
    if d["degenerate"]:
        print("    NOTE: one radius ≈ 0 — degenerate (a great circle in one "
              "2-plane), not a genuine 2-torus.")
    verdict = ("ON a Clifford torus" if d["on_clifford"] and not d["degenerate"]
               else "great circle (degenerate)" if d["degenerate"]
               else "OFF the Clifford torus")
    rr = max(d["r1"], d["r2"])
    print("    VERDICT            : %s%s"
          % (verdict, ("   radius r ≈ %.4f" % rr)
             if d["on_clifford"] and not d["degenerate"] else ""))


def analytic_clifford_trefoil(r, n=2000):
    """(2,3) torus knot on T_r in S^3, the positive control."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    s = np.sqrt(max(0.0, 1.0 - r * r))
    return np.column_stack([r * np.cos(2 * th), r * np.sin(2 * th),
                            s * np.cos(3 * th), s * np.sin(3 * th)])


def selftest():
    print("SELFTEST — analytic controls")
    print("=" * 70)
    for r in (0.776246, 0.5, 0.95):
        x = analytic_clifford_trefoil(r, 2000)
        report("analytic Clifford trefoil  r=%.4f (expect ON, radius=%.4f)"
               % (r, r), clifford_diagnostic(x))
        print()
    # a great circle (degenerate 1-plane) should NOT read as a 2-torus
    th = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    gc = np.column_stack([np.cos(th), np.sin(th), 0 * th, 0 * th])
    report("great circle (lies in a single 2-plane, not a 2-torus)",
           clifford_diagnostic(gc))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vects", nargs="*", help="minimised _s3.vect file(s)")
    ap.add_argument("--cv-thresh", type=float, default=0.05,
                    help="radius CV below which it counts as constant (default 0.05)")
    ap.add_argument("--selftest", action="store_true",
                    help="run analytic controls instead of files")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.vects:
        ap.error("give one or more _s3.vect files, or --selftest")

    print("Clifford-torus check (S^3 covariance / constant-radius test)")
    print("=" * 70)
    for path in args.vects:
        y = read_vect_r3(path)
        x = lift_to_s3(y)
        report(os.path.relpath(path), clifford_diagnostic(x, args.cv_thresh))
        print()


if __name__ == "__main__":
    main()
