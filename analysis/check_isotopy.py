#!/usr/bin/env python3
"""
check_isotopy.py — verify that a .vect knot is T(2,3) via pyknotid invariants.

Usage:
    python3 analysis/check_isotopy.py <path.vect>

Exit codes:
    0 — PASS  (all invariants match T(2,3))
    1 — FAIL  (at least one invariant mismatch or error)

T(2,3) trefoil expected values
    determinant   : 3
    vassiliev_2   : 1
    vassiliev_3   : 1
"""

import sys
import os
import argparse

import numpy as np

# ── numpy compatibility shim for pyknotid ──────────────────────────────────
# pyknotid was written against older numpy that exposed np.float etc. as top-
# level names.  Patch them back before importing pyknotid so nothing crashes.
np.float   = float          # type: ignore[attr-defined]
np.int     = int            # type: ignore[attr-defined]
np.complex = complex        # type: ignore[attr-defined]
np.long    = np.int64       # type: ignore[attr-defined]

try:
    from pyknotid.spacecurves import Knot
except ImportError:
    print("Error: pyknotid not installed.  Run: pip3 install pyknotid",
          file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Expected invariants for T(2,3)
# ═══════════════════════════════════════════════════════════════════════════

EXPECTED = {
    "determinant":  3,
    "vassiliev_2":  1,
    "vassiliev_3":  1,
}

KNOT_NAME = "T(2,3)  (trefoil  3₁)"


# ═══════════════════════════════════════════════════════════════════════════
# .vect reader
# ═══════════════════════════════════════════════════════════════════════════

def load_vect(path: str) -> np.ndarray:
    """
    Read a .vect file.  Format:
        <n_components>
        <n_points>
        x y z
        ...
    Returns (n_points, 3) float64 array.
    """
    with open(path) as f:
        n_comp  = int(f.readline())
        n_pts   = int(f.readline())
        pts = np.fromiter(
            (float(v) for line in f for v in line.split()),
            dtype=np.float64,
        ).reshape(-1, 3)
    if len(pts) != n_pts:
        raise ValueError(
            f"Header says {n_pts} points but read {len(pts)}"
        )
    if n_comp != 1:
        print(f"Warning: {n_comp} components; using first {n_pts} points.",
              file=sys.stderr)
    return pts


# ═══════════════════════════════════════════════════════════════════════════
# Invariant computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_invariants(pts: np.ndarray) -> dict:
    """Build a pyknotid Knot and compute the required invariants."""
    k = Knot(pts, verbose=False)

    results = {}

    # Number of crossings in the Gauss code projection
    try:
        gc = k.gauss_code()
        results["crossings"] = len(gc) // 2 if gc else 0
    except Exception as e:
        results["crossings"] = f"Error: {e}"

    # Determinant  (|Δ(-1)| = |Alexander polynomial at t=-1|)
    try:
        results["determinant"] = int(abs(k.determinant()))
    except Exception as e:
        results["determinant"] = f"Error: {e}"

    # Vassiliev degree-2 invariant  (= coefficient c₂ of Conway polynomial)
    try:
        results["vassiliev_2"] = int(k.vassiliev_degree_2())
    except Exception as e:
        results["vassiliev_2"] = f"Error: {e}"

    # Vassiliev degree-3 invariant
    try:
        results["vassiliev_3"] = int(k.vassiliev_degree_3())
    except Exception as e:
        results["vassiliev_3"] = f"Error: {e}"

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Pass / fail check
# ═══════════════════════════════════════════════════════════════════════════

def check(results: dict) -> bool:
    passed = True
    for key, expected_val in EXPECTED.items():
        got = results.get(key, "missing")
        if isinstance(got, str):          # error string
            passed = False
        elif abs(int(got)) != abs(int(expected_val)):
            passed = False
    return passed


# ═══════════════════════════════════════════════════════════════════════════
# Pretty output
# ═══════════════════════════════════════════════════════════════════════════

def print_results(results: dict, passed: bool, vect_path: str) -> None:
    width = 42
    bar   = "═" * width

    print(f"\n{bar}")
    print(f"  File  : {os.path.basename(vect_path)}")
    print(f"  Knot  : {KNOT_NAME}")
    print(f"  ─────────────────────────────────────")

    label_w = 18
    for key in ("crossings", "determinant", "vassiliev_2", "vassiliev_3"):
        got      = results.get(key, "—")
        expected = EXPECTED.get(key)
        if expected is None:
            tag = ""
        else:
            ok  = (not isinstance(got, str)) and abs(int(got)) == abs(int(expected))
            tag = "  ✓" if ok else f"  ✗  (expected {expected})"
        print(f"  {key:<{label_w}}: {got}{tag}")

    print(f"  ─────────────────────────────────────")
    if passed:
        print(f"  Isotopy check: PASS ✓")
        print(f"  Knot identified as T(2,3)")
    else:
        print(f"  Isotopy check: FAIL ✗")
        print(f"  One or more invariants do not match T(2,3)")
    print(f"{bar}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that a .vect knot is T(2,3) via pyknotid invariants"
    )
    parser.add_argument("vect", help="Input .vect file")
    args = parser.parse_args()

    if not os.path.exists(args.vect):
        print(f"Error: {args.vect} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.vect} …")
    pts = load_vect(args.vect)
    print(f"  {len(pts)} vertices")

    print("Computing pyknotid invariants …")
    results = compute_invariants(pts)

    passed = check(results)
    print_results(results, passed, args.vect)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
