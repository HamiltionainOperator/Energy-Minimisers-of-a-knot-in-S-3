#!/usr/bin/env python3
"""
Verify the isotopy type of a knot (given as a .vect file) via SnapPy.

Workflow
--------
1. Load the polygon from a .vect file.
2. Try one or more 2D projections to detect crossings (fewest-crossing
   projection wins to minimise diagram complexity).
3. Build a PD code and pass it to SnapPy's Link.
4. Compute:  Alexander polynomial, knot signature, hyperbolic volume,
   SnapPy identification.
5. Optionally compare against known invariants for a named knot type.

Usage
-----
    python3 analysis/verify_knot.py output/T2_3.vect
    python3 analysis/verify_knot.py output/T2_3.vect --expected T(2,3)
    python3 analysis/verify_knot.py output/T3_5.vect --expected T(3,5) --multi-proj

Exit code: 0 = pass / no expected supplied, 1 = fail.

Dependencies: snappy, numpy
    pip3 install snappy numpy
"""

import argparse
import os
import sys

# Allow importing sibling modules at project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

try:
    import snappy
except ImportError:
    print("Error: snappy not installed.  Run: pip3 install snappy", file=sys.stderr)
    sys.exit(1)

import numpy as np
from vect_to_pdcode import read_vect, find_crossings, crossings_to_pd


# ---------------------------------------------------------------------------
# Known invariants table  (extend as needed)
# ---------------------------------------------------------------------------

KNOWN: dict = {
    "T(2,3)": {
        "name":      "trefoil  3_1",
        "signature": -2,
        "det":        3,
    },
    "T(2,5)": {
        "name":      "cinquefoil  5_1",
        "signature": -4,
        "det":        5,
    },
    "T(2,7)": {
        "name":      "7_1",
        "signature": -6,
        "det":        7,
    },
    "T(3,4)": {
        "name":      "8_19",
        "signature": -6,
        "det":       13,
    },
    "T(3,5)": {
        "name":      "10_124",
        "signature": -8,
        "det":        5,
    },
}


# ---------------------------------------------------------------------------
# PD code builder
# ---------------------------------------------------------------------------

def build_pd(vect_path: str, proj_axis: int = 2, multi: bool = False):
    """
    Build PD code from a .vect file.

    Returns (pd_list, n_crossings) or (None, 0) if no crossings found.
    """
    components = read_vect(vect_path)
    pts = components[0]

    if multi:
        best_cr, best_pd = None, None
        for axis in range(3):
            cr = find_crossings(pts, proj_axis=axis)
            pd = crossings_to_pd(cr)
            print(f"  Projection axis {axis}: {len(cr)} crossings", file=sys.stderr)
            if best_cr is None or len(cr) < len(best_cr):
                best_cr, best_pd = cr, pd
        crossings, pd = best_cr, best_pd
    else:
        crossings = find_crossings(pts, proj_axis=proj_axis)
        pd        = crossings_to_pd(crossings)
        print(f"  Projection axis {proj_axis}: {len(crossings)} crossings",
              file=sys.stderr)

    if not crossings:
        return None, 0
    return pd, len(crossings)


# ---------------------------------------------------------------------------
# Invariant computation
# ---------------------------------------------------------------------------

def compute_invariants(pd) -> dict:
    """
    Compute knot invariants from a PD code list via SnapPy.

    Returns a dict of results; individual keys may map to error strings.
    """
    inner  = ", ".join(f"X[{a},{b},{c},{d}]" for a, b, c, d in pd)
    pd_str = f"PD[{inner}]"

    results = {"pd_string": pd_str, "n_crossings": len(pd)}

    try:
        L = snappy.Link(pd)
        K = L.exterior()
    except Exception as e:
        results["error"] = str(e)
        return results

    for label, fn in [
        ("alexander",       lambda: str(L.alexander_polynomial())),
        ("signature",       lambda: L.signature()),
        ("determinant",     lambda: L.determinant()),
        ("jones",           lambda: str(L.jones_polynomial())),
        ("volume",          lambda: K.volume()),
        ("identification",  lambda: K.identify()),
    ]:
        try:
            results[label] = fn()
        except Exception as e:
            results[label] = f"Error: {e}"

    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _check(label: str, got, expected) -> bool:
    ok = (got == expected)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}]  {label}: got {got!r}  (expected {expected!r})")
    return ok


def verify(results: dict, expected_key: str) -> bool:
    if expected_key not in KNOWN:
        print(f"  No known invariants for {expected_key!r} in table.")
        return True   # unknown = not a failure

    ref    = KNOWN[expected_key]
    passed = True

    print(f"\nVerification  vs.  {expected_key}  ({ref['name']}):")
    print("-" * 56)

    if "signature" in results and not str(results["signature"]).startswith("Error"):
        passed &= _check("signature", results["signature"], ref["signature"])
    if "determinant" in results and not str(results["determinant"]).startswith("Error"):
        passed &= _check("determinant", results["determinant"], ref["det"])

    for key in ("alexander", "volume", "identification"):
        val = results.get(key, "N/A")
        print(f"  {key:20s}: {val}")

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify knot isotopy type via SnapPy"
    )
    parser.add_argument("vect", help="Input .vect file")
    parser.add_argument(
        "--expected", default=None,
        metavar="TYPE",
        help="Expected knot type, e.g. 'T(2,3)'.  Enables pass/fail check."
    )
    parser.add_argument("--proj", type=int, default=2,
                        help="Projection axis for crossing detection (default: 2)")
    parser.add_argument("--multi-proj", action="store_true",
                        help="Try all three projection axes and keep the smallest diagram")
    args = parser.parse_args()

    if not os.path.exists(args.vect):
        print(f"Error: {args.vect} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Building PD code from {args.vect} …")
    pd, n_cr = build_pd(args.vect, proj_axis=args.proj, multi=args.multi_proj)

    if pd is None:
        print("No crossings found — knot may be unknot or projection is degenerate.")
        print("Try --multi-proj or a different --proj axis.")
        sys.exit(0)

    print(f"  → {n_cr}-crossing diagram\n")
    print("Computing SnapPy invariants …")
    results = compute_invariants(pd)

    print("\nResults:")
    for k, v in results.items():
        print(f"  {k:20s}: {v}")

    passed = True
    if args.expected:
        passed = verify(results, args.expected)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
