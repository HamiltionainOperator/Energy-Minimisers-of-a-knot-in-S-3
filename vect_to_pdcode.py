#!/usr/bin/env python3
"""
Convert a .vect polygonal-curve file to PD (planar diagram) code for SnapPy.

Algorithm
---------
1. Read the R³ polygon from the .vect file.
2. Project onto a 2D plane (default: drop z-axis) and detect crossings of
   line-segment pairs using exact rational arithmetic via integer arithmetic
   on floats (with epsilon tolerance).
3. Walk the curve in parameter order, labelling arcs between consecutive
   crossing events.  Each crossing receives four arc labels in standard PD
   convention: X[a, b, c, d] where
       a = incoming under,  b = outgoing over,
       c = outgoing under,  d = incoming over.
4. Emit a SnapPy-ready Python snippet and optionally write it to a file.

Dependencies: numpy (scipy optional, used for PD normalisation)

Usage:
    python3 vect_to_pdcode.py knot.vect                     # print to stdout
    python3 vect_to_pdcode.py knot.vect --out knot_pd.py    # write to file
    python3 vect_to_pdcode.py knot.vect --multi-proj        # try all 3 axes
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# .vect reader
# ---------------------------------------------------------------------------

def read_vect(path: str) -> List[np.ndarray]:
    """
    Read a .vect file.  Returns a list of (N_i, 3) arrays, one per component.
    """
    components = []
    with open(path) as f:
        n_comp = int(f.readline().strip())
        for _ in range(n_comp):
            n_pts = int(f.readline().strip())
            pts   = []
            for _ in range(n_pts):
                coords = list(map(float, f.readline().split()))
                pts.append(coords)
            components.append(np.array(pts, dtype=float))
    return components


# ---------------------------------------------------------------------------
# 2-D segment intersection
# ---------------------------------------------------------------------------

def _seg_intersect_2d(
    a1: np.ndarray, a2: np.ndarray,
    b1: np.ndarray, b2: np.ndarray,
    eps: float = 1e-9,
) -> Optional[Tuple[float, float]]:
    """
    Compute interior intersection parameters (ta, tb) ∈ (0,1)² of segments
    a1→a2 and b1→b2 in ℝ².  Returns None if they do not properly intersect.
    """
    d1 = a2 - a1
    d2 = b2 - b1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < eps:
        return None   # parallel / collinear
    diff = b1 - a1
    ta = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    tb = (diff[0] * d1[1] - diff[1] * d1[0]) / denom
    if eps < ta < 1 - eps and eps < tb < 1 - eps:
        return ta, tb
    return None


# ---------------------------------------------------------------------------
# Crossing detection
# ---------------------------------------------------------------------------

def find_crossings(pts: np.ndarray, proj_axis: int = 2) -> List[dict]:
    """
    Find all crossings of the closed polygon *pts* projected along *proj_axis*.

    Returns a list of crossing dicts with keys:
        edge_i, ta   — index + parameter of first edge
        edge_j, tb   — index + parameter of second edge
        over         — index of the over-strand edge
        sign         — crossing sign (+1 or -1)
        param_i      — float parameter along curve for edge_i event
        param_j      — float parameter along curve for edge_j event
    """
    n = len(pts)
    axes    = [i for i in range(3) if i != proj_axis]
    pts2d   = pts[:, axes]
    height  = pts[:, proj_axis]

    crossings = []
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue   # adjacent edges share a vertex
            j2 = (j + 1) % n
            res = _seg_intersect_2d(pts2d[i], pts2d[i2], pts2d[j], pts2d[j2])
            if res is None:
                continue
            ta, tb = res
            h_i = (1 - ta) * height[i]  + ta * height[i2]
            h_j = (1 - tb) * height[j]  + tb * height[j2]
            over  = i if h_i > h_j else j
            under = j if h_i > h_j else i

            # Crossing sign: right-hand rule between over-strand and
            # under-strand tangents in the projection plane
            d_over  = pts2d[(over  + 1) % n] - pts2d[over]
            d_under = pts2d[(under + 1) % n] - pts2d[under]
            sign = +1 if d_over[0] * d_under[1] - d_over[1] * d_under[0] > 0 else -1

            crossings.append({
                "edge_i": i,  "ta": ta,  "param_i": i + ta,
                "edge_j": j,  "tb": tb,  "param_j": j + tb,
                "over":  over,  "sign": sign,
            })

    return crossings


# ---------------------------------------------------------------------------
# Arc labelling → PD code
# ---------------------------------------------------------------------------

def crossings_to_pd(crossings: List[dict]) -> List[Tuple[int, int, int, int]]:
    """
    Walk the curve and assign arc labels, then build the PD code.

    Arc labels are integers 1 … 2·|crossings|.

    PD convention: X[a, b, c, d]
        a = incoming under,  b = outgoing over,
        c = outgoing under,  d = incoming over
    """
    nc = len(crossings)
    if nc == 0:
        return []

    # Collect all crossing events sorted by their parameter along the curve
    events = []
    for ci, c in enumerate(crossings):
        events.append((c["param_i"], ci, "i"))
        events.append((c["param_j"], ci, "j"))
    events.sort(key=lambda e: e[0])

    n_arcs      = 2 * nc
    arc_counter = 1                  # arcs labelled 1 .. 2*nc
    # crossing_info[ci] = {"i_in", "i_out", "j_in", "j_out"}
    crossing_info: List[dict] = [{} for _ in range(nc)]

    for (_, ci, which) in events:
        crossing_info[ci][f"{which}_in"]  = arc_counter
        arc_counter_next = arc_counter % n_arcs + 1
        crossing_info[ci][f"{which}_out"] = arc_counter_next
        arc_counter = arc_counter_next

    pd = []
    for ci, c in enumerate(crossings):
        info = crossing_info[ci]
        over_edge = c["over"]
        if over_edge == c["edge_i"]:
            # edge i is over
            a = info.get("j_in",  1)   # incoming under
            b = info.get("i_out", 2)   # outgoing over
            cc= info.get("j_out", 3)   # outgoing under
            d = info.get("i_in",  4)   # incoming over
        else:
            # edge j is over
            a = info.get("i_in",  1)
            b = info.get("j_out", 2)
            cc= info.get("i_out", 3)
            d = info.get("j_in",  4)
        pd.append((a, b, cc, d))

    return pd


# ---------------------------------------------------------------------------
# SnapPy output
# ---------------------------------------------------------------------------

def pd_to_snappy(pd: List[Tuple[int, int, int, int]], p: int, q: int) -> str:
    """
    Format PD code as a self-contained SnapPy Python snippet.
    """
    inner = ", ".join(f"X[{a},{b},{c},{d}]" for a, b, c, d in pd)
    pd_str = f"PD[{inner}]"
    lines = [
        "import snappy",
        f"# {len(pd)}-crossing PD code  (T({p},{q}) if input was a torus knot)",
        f"L = snappy.Link('{pd_str}')",
        "K = L.exterior()",
        "print('Alexander polynomial:', L.alexander_polynomial())",
        "print('Signature:',            L.signature())",
        "print('Volume:',               K.volume())",
        "print('Identification:',       K.identify())",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert .vect to SnapPy PD code"
    )
    parser.add_argument("input", help="Input .vect file")
    parser.add_argument("--proj", type=int, default=2,
                        help="Projection axis to drop  0=x, 1=y, 2=z (default: 2)")
    parser.add_argument("--multi-proj", action="store_true",
                        help="Try all three projection axes and keep fewest crossings")
    parser.add_argument("--out", default=None,
                        help="Output .py file (default: stdout)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    components = read_vect(args.input)
    if len(components) != 1:
        print(f"Warning: {len(components)} components — using component 0 only.",
              file=sys.stderr)
    pts = components[0]

    # Guess p and q from filename if possible (e.g. "T2_3.vect")
    base = os.path.splitext(os.path.basename(args.input))[0]
    try:
        _, pq = base.split("T", 1)
        p_kn, q_kn = (int(x) for x in pq.split("_"))
    except Exception:
        p_kn, q_kn = 0, 0

    if args.multi_proj:
        best_pd  = None
        best_n   = float("inf")
        for axis in range(3):
            cr  = find_crossings(pts, proj_axis=axis)
            print(f"Axis {axis}: {len(cr)} crossings", file=sys.stderr)
            pd  = crossings_to_pd(cr)
            if pd is not None and len(cr) < best_n:
                best_pd, best_n = pd, len(cr)
        pd = best_pd or []
    else:
        crossings = find_crossings(pts, proj_axis=args.proj)
        print(f"Found {len(crossings)} crossings (proj axis={args.proj})",
              file=sys.stderr)
        pd = crossings_to_pd(crossings)

    if not pd:
        out_str = (
            "import snappy\n"
            "# No crossings detected — likely unknot or degenerate projection.\n"
            "L = snappy.Link('[]')  # unknot\n"
        )
    else:
        out_str = pd_to_snappy(pd, p_kn, q_kn)

    if args.out:
        with open(args.out, "w") as f:
            f.write(out_str)
        print(f"Written to {args.out}")
    else:
        print(out_str)


if __name__ == "__main__":
    main()
