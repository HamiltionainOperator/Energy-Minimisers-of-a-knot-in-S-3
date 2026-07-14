#!/usr/bin/env python3
"""
Genus-formula scan for O'Hara's SPHERICAL QUANTITY energy  E_q = L^-2 · E_{S^3}.

For each knot (a torus T(p,q) or a connect-sum), this:
  1. generates it,
  2. minimises E_q with build/energy_s3 --energy quantity,
  3. records the minimum E_q (the min over the whole energy_log, not just the
     last iter), the Seifert genus, and the robust determinant before AND after
     minimisation (so a topology change during the flow is caught), and
  4. prints a table of  min E_q  vs  genus  so you can see whether E_q scales
     with genus the way the density energy does (E_d,min(T(p,q)) ~ 54.31·genus).

Genus / topology facts used:
  torus T(p,q):  g = (p-1)(q-1)/2
  connect-sum :  genus is ADDITIVE,  g(K1 # K2 # ...) = sum g(Ki)
  determinant of a connect-sum = product of the components' determinants.
We don't hard-code component determinants: we compare the robust determinant of
the MINIMISED curve against that of the freshly GENERATED curve (which the
generators are validated to produce correctly) — equal => knot type preserved.

Knot spec syntax (positional args):
  "2,3"          a torus knot T(2,3)
  "2,3#2,3"      a connect-sum  T(2,3) # T(2,3)  (any number of #-joined parts)

Usage
-----
  python3 analysis/genus_scan_quantity.py                          # default batch
  python3 analysis/genus_scan_quantity.py 2,3 2,5 2,7 2,3#2,3      # custom list
  python3 analysis/genus_scan_quantity.py --n 1500 --iters 6000 2,3 2,3#2,5
  python3 analysis/genus_scan_quantity.py --step 0.01 --out output/_qscan 2,3 3,4
"""
import argparse
import csv
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "energy_s3")
PYEXE = sys.executable


def parse_spec(spec):
    """'2,3#2,5' -> [(2,3),(2,5)] ; '2,3' -> [(2,3)]."""
    parts = []
    for token in spec.split("#"):
        p, q = (int(x) for x in token.split(","))
        parts.append((p, q))
    return parts


def torus_genus(p, q):
    return (p - 1) * (q - 1) // 2


def robust_det(path, ntries=11):
    """Robust determinant via majority vote over random projections."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from knot_check import read_vect_pts, robust_determinant
    pts = read_vect_pts(path)
    det, dist, unanimous = robust_determinant(pts, ntries=ntries)
    return det, unanimous


def min_energy_from_log(log_path):
    """Minimum of the 'energy' column over the whole run."""
    best = None
    with open(log_path) as f:
        for row in csv.DictReader(f):
            e = float(row["energy"])
            best = e if best is None else min(best, e)
    return best


def length_range_from_log(log_path):
    """(L0, L_final) from the total_length column, if present."""
    L0 = Lf = None
    with open(log_path) as f:
        for row in csv.DictReader(f):
            if "total_length" not in row:
                return (None, None)
            L = float(row["total_length"])
            if L0 is None:
                L0 = L
            Lf = L
    return (L0, Lf)


def run_one(spec, out_root, n, iters, step):
    parts = parse_spec(spec)
    is_torus = len(parts) == 1
    genus = sum(torus_genus(p, q) for (p, q) in parts)
    name = "T%d_%d" % parts[0] if is_torus else \
        "cs_" + "_".join("%dx%d" % pq for pq in parts)
    out_dir = os.path.join(out_root, name + "_q")
    os.makedirs(out_dir, exist_ok=True)
    init_vect = os.path.join(out_dir, name + ".vect")
    s3_vect = os.path.join(out_dir, name + "_s3.vect")
    log = os.path.join(out_dir, "energy_log.csv")

    # 1) generate
    if is_torus:
        p, q = parts[0]
        subprocess.run([PYEXE, os.path.join(ROOT, "knots", "generate.py"),
                        str(p), str(q), "--n", str(n), "--out", out_dir],
                       check=True, capture_output=True, text=True)
        norm = ["--no-normalize"]        # torus knots start on the Clifford torus
    else:
        connect = " ".join("%d,%d" % pq for pq in parts)
        subprocess.run([PYEXE, os.path.join(ROOT, "knots", "generate_composites.py"),
                        "--connect", *connect.split(), "--n", str(n),
                        "--out", init_vect], check=True, capture_output=True, text=True)
        norm = []                        # composites are placed in R3 -> normalise

    det0, uni0 = robust_det(init_vect)

    # 2) minimise E_q
    cmd = [BIN, init_vect, log, s3_vect, str(iters), str(step),
           "--energy", "quantity"] + norm
    proc = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"Final\s+E_q energy\s*:\s*([0-9.eE+-]+)", proc.stdout)
    e_final = float(m.group(1)) if m else float("nan")
    e_min = min_energy_from_log(log)
    L0, Lf = length_range_from_log(log)

    det1, uni1 = robust_det(s3_vect)
    held = (det0 == det1)

    return {
        "name": name, "spec": spec, "genus": genus, "is_torus": is_torus,
        "e_min": e_min, "e_final": e_final,
        "det0": det0, "det1": det1, "held": held,
        "uni": uni0 and uni1, "L0": L0, "Lf": Lf,
    }


DEFAULT_BATCH = ["2,3", "2,5", "2,7", "2,3#2,3", "2,3#2,5"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("knots", nargs="*", default=DEFAULT_BATCH,
                    help="knot specs, e.g. 2,3  2,5  2,3#2,3  (default: a 5-knot batch)")
    ap.add_argument("--n", type=int, default=1000, help="vertices (default 1000)")
    ap.add_argument("--iters", type=int, default=4000, help="max iterations (default 4000)")
    ap.add_argument("--step", type=float, default=0.01, help="initial alpha0 (default 0.01)")
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "_qscan"),
                    help="output root (default output/_qscan)")
    ap.add_argument("--md", default=None, help="also write a markdown table here")
    args = ap.parse_args()

    knots = args.knots if args.knots else DEFAULT_BATCH
    print("Genus-formula scan — QUANTITY energy E_q = L^-2 E_{S^3}")
    print("  N=%d  iters=%d  step=%g" % (args.n, args.iters, args.step))
    print("=" * 78)

    rows = []
    for spec in knots:
        print("  running %-14s ..." % spec, end="", flush=True)
        try:
            r = run_one(spec, args.out, args.n, args.iters, args.step)
            rows.append(r)
            print("  min E_q=%.5f  g=%d  det %s->%s  %s"
                  % (r["e_min"], r["genus"], r["det0"], r["det1"],
                     "HELD" if r["held"] else "CHANGED!"))
        except Exception as exc:
            print("  ERROR: %s" % exc)

    if not rows:
        return

    # reference = the smallest-genus torus knot present (usually T(2,3))
    ref = min((r for r in rows), key=lambda r: (r["genus"], not r["is_torus"]))
    e_ref, g_ref = ref["e_min"], max(1, ref["genus"])

    print("\n%-12s %-8s %5s %12s %12s %10s %8s" %
          ("knot", "kind", "g", "min E_q", "E_q/g", "E_q/E_ref", "det"))
    print("-" * 78)
    for r in sorted(rows, key=lambda r: (r["genus"], not r["is_torus"])):
        print("%-12s %-8s %5d %12.5f %12.5f %10.3f %8s" %
              (r["name"], "torus" if r["is_torus"] else "comp",
               r["genus"], r["e_min"], r["e_min"] / max(1, r["genus"]),
               r["e_min"] / e_ref, "%s%s" % (r["det1"], "" if r["held"] else "*")))
    print("-" * 78)
    print("ref = %s (g=%d, E_q=%.5f).  E_q/g constant <=> linear genus law." %
          (ref["name"], ref["genus"], e_ref))
    print("'*' on det = knot type changed during the flow (result not trustworthy).")

    if args.md:
        with open(args.md, "w") as f:
            f.write("# Quantity-energy genus scan (E_q = L^-2 E_{S^3})\n\n")
            f.write("N=%d, iters=%d, step=%g\n\n" % (args.n, args.iters, args.step))
            f.write("| knot | kind | genus | min E_q | E_q/g | E_q/E_ref | det | held |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in sorted(rows, key=lambda r: (r["genus"], not r["is_torus"])):
                f.write("| %s | %s | %d | %.5f | %.5f | %.3f | %s | %s |\n" %
                        (r["name"], "torus" if r["is_torus"] else "comp",
                         r["genus"], r["e_min"], r["e_min"] / max(1, r["genus"]),
                         r["e_min"] / e_ref, r["det1"], "yes" if r["held"] else "NO"))
        print("\nmarkdown -> %s" % args.md)


if __name__ == "__main__":
    main()
