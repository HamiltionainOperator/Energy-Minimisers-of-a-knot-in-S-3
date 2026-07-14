#!/usr/bin/env python3
"""
Re-evaluate already-converged knot configs under BOTH ambient energies.

Two energies, ONE shared evaluator (build/energy_s3): the geodesic S³ O'Hara
energy and the chordal-Euclidean Möbius energy.  They differ only in the ambient
pairwise distance —

    geodesic:  d_S³(x,y)      = arccos⟨x,y⟩
    chordal :  |x−y|_{ℝ⁴}     = 2 sin(d_S³/2)

— while the intrinsic arc-length distance d_K, the arc-length weights, the
diagonal cutoff and the quadrature are identical (see compute_energy_ohara in
Repulsor/energy_s3.cpp; the C++ does the math, this script only orchestrates and
tabulates, so there is no second implementation of the energy).

This does NO minimisation: it re-evaluates the existing converged curves so the
geodesic↔chordal comparison can be eyeballed before deciding whether a full
Euclidean minimisation is worth running.

Usage
-----
    python3 analysis/eval_dual_energy.py                 # auto-discover configs
    python3 analysis/eval_dual_energy.py output/T2_3/T2_3_s3.vect ...
    python3 analysis/eval_dual_energy.py --binary build/energy_s3
"""
import argparse
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# DUAL <tag> n=<n> E_geo=<f> E_chordal=<f> ratio=<f|nan>
_DUAL = re.compile(
    r"DUAL\s+(?P<tag>\S+)\s+n=(?P<n>\d+)\s+E_geo=(?P<geo>[-\d.eE+]+)\s+"
    r"E_chordal=(?P<chd>[-\d.eE+]+)\s+ratio=(?P<ratio>\S+)"
)

# Converged torus-knot config:  output/T<p>_<q>/T<p>_<q>_s3.vect  (no suffix
# variants like _ref / _rw7, no composites).
_TORUS_DIR = re.compile(r"^T(\d+)_(\d+)$")


def torus_genus(p: int, q: int) -> int:
    """Seifert genus of the torus knot T(p,q):  (p−1)(q−1)/2."""
    return (p - 1) * (q - 1) // 2


def discover_configs():
    """Find converged torus configs, return [(label, p, q, genus, path)]."""
    found = []
    for path in glob.glob(os.path.join(ROOT, "output", "*", "*_s3.vect")):
        d = os.path.basename(os.path.dirname(path))
        m = _TORUS_DIR.match(d)
        if not m:
            continue
        p, q = int(m.group(1)), int(m.group(2))
        if os.path.basename(path) != f"{d}_s3.vect":
            continue
        found.append((f"T({p},{q})", p, q, torus_genus(p, q), path))
    found.sort(key=lambda r: (r[1], r[2]))
    return found


def run_eval(binary: str, paths):
    """Call the binary once on all paths; return {path: dict(parsed fields)}."""
    proc = subprocess.run(
        [binary, "--eval", *paths], capture_output=True, text=True
    )
    out = {}
    for line in (proc.stdout + proc.stderr).splitlines():
        m = _DUAL.search(line)
        if m:
            out[m.group("tag")] = {
                "n": int(m.group("n")),
                "geo": float(m.group("geo")),
                "chd": float(m.group("chd")),
            }
    if not out and proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
    return out


def run_calibration(binary: str, n: int = 800):
    proc = subprocess.run(
        [binary, "--calibrate", str(n)], capture_output=True, text=True
    )
    for line in (proc.stdout + proc.stderr).splitlines():
        m = _DUAL.search(line)
        if m:
            return float(m.group("geo")), float(m.group("chd")), int(m.group("n"))
    sys.stderr.write(proc.stdout + proc.stderr)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*",
                    help="explicit *_s3.vect paths (default: auto-discover torus configs)")
    ap.add_argument("--binary", default=os.path.join(ROOT, "build", "energy_s3"))
    ap.add_argument("--calib-n", type=int, default=800)
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit(f"binary not found: {args.binary}  (run `make build`)")

    # ── Calibration: round great-circle unknot ─────────────────────────────
    print("Calibration — round great-circle unknot")
    print("─" * 52)
    cal = run_calibration(args.binary, args.calib_n)
    if cal:
        geo, chd, n = cal
        print(f"  n            : {n}")
        print(f"  E_geodesic   : {geo:.6g}   (expect ≈ 0; great circle is an "
              f"S³ geodesic ⇒ d_S³ = d_K)")
        print(f"  E_chordal    : {chd:.6g}   (round-circle Möbius value, → 4 as "
              f"n → ∞)")
    print()

    # ── Re-evaluate converged configs ──────────────────────────────────────
    if args.configs:
        rows_meta = []
        for p in args.configs:
            d = os.path.basename(os.path.dirname(p))
            m = _TORUS_DIR.match(d)
            if m:
                pp, qq = int(m.group(1)), int(m.group(2))
                rows_meta.append((f"T({pp},{qq})", pp, qq, torus_genus(pp, qq), p))
            else:
                rows_meta.append((d, None, None, None, p))
    else:
        rows_meta = discover_configs()

    if not rows_meta:
        sys.exit("No configs found to evaluate.")

    paths = [r[4] for r in rows_meta]
    data = run_eval(args.binary, paths)

    print(f"Re-evaluation of {len(rows_meta)} converged config(s) "
          f"(no re-minimisation)")
    print("─" * 72)
    hdr = f"{'knot':<10}{'g':>3}  {'n':>6}  {'E_geo':>13}  {'E_chordal':>13}  {'ratio':>8}"
    print(hdr)
    print("─" * 72)
    for label, p, q, g, path in rows_meta:
        rec = data.get(path)
        gs = str(g) if g is not None else "–"
        if not rec:
            print(f"{label:<10}{gs:>3}  {'?':>6}  {'(eval failed)':>13}")
            continue
        ratio = rec["chd"] / rec["geo"] if abs(rec["geo"]) > 1e-9 else float("nan")
        print(f"{label:<10}{gs:>3}  {rec['n']:>6}  {rec['geo']:>13.6f}  "
              f"{rec['chd']:>13.6f}  {ratio:>8.4f}")


if __name__ == "__main__":
    main()
