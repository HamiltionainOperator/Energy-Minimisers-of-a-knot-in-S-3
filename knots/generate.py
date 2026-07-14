#!/usr/bin/env python3
"""
Generate a (p,q)-torus knot on S³ and write .obj, .vect, and scene.txt.

Canonical Clifford-torus embedding of T(p,q) in S³ ⊂ ℝ⁴:

    γ(t) = ( cos(p·t)/√2,  sin(p·t)/√2,
             cos(q·t)/√2,  sin(q·t)/√2 ),   t ∈ [0, 2π)

Stereographic projection from north pole N = (0,0,0,1):

    π(x₁,x₂,x₃,x₄) = (x₁, x₂, x₃) / (1 − x₄)

Usage:
    python3 knots/generate.py 2 3
    python3 knots/generate.py 3 5 --n 400 --out output/T3_5
"""

import argparse
import math
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _torus_knot_s3_uniform_t(p: int, q: int, m: int) -> np.ndarray:
    """(m, 4) unit points on S³ for T(p,q), sampled uniformly in the parameter t."""
    t = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return np.column_stack([
        inv_sqrt2 * np.cos(p * t),
        inv_sqrt2 * np.sin(p * t),
        inv_sqrt2 * np.cos(q * t),
        inv_sqrt2 * np.sin(q * t),
    ])


def torus_knot_s3(p: int, q: int, n_points: int = 200,
                  oversample: int = 64) -> np.ndarray:
    """Return (n_points, 4) unit points on S³ parametrising T(p,q), spaced at
    EQUAL geodesic arc length on S³.

    Sampling uniformly in the parameter t clusters vertices wherever the curve
    moves slowly (the q-direction wobble of T(2,q) is a strong example), and the
    stereographic image distorts this further — so a uniform-t initialisation
    looks lopsided, with one lobe point-starved and another over-dense. That
    asymmetry then biases the early energy flow and the diagnostics. We instead
    oversample densely in t, measure the cumulative geodesic length on S³, and
    re-pick n_points at equal arc length (slerping along each dense geodesic
    edge). This is exactly the parametrisation reparametrize_s3() maintains in
    the C++ optimiser, so the knot now STARTS uniform rather than being dragged
    there over the first ~50 iterations.
    """
    m = max(n_points * oversample, n_points)
    dense = _torus_knot_s3_uniform_t(p, q, m)              # (m,4) on S³

    # Geodesic length of each closed edge:  θ_k = arccos⟨x_k, x_{k+1}⟩.
    cdot = np.clip(np.einsum("ij,ij->i", dense, np.roll(dense, -1, axis=0)),
                   -1.0, 1.0)
    seg = np.arccos(cdot)
    cum = np.concatenate([[0.0], np.cumsum(seg)])           # (m+1,)
    total = cum[-1]

    targets = np.linspace(0.0, total, n_points, endpoint=False)
    j = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, m - 1)
    th = seg[j]
    f = np.where(th > 1e-12, (targets - cum[j]) / np.where(th > 1e-12, th, 1.0), 0.0)

    a = dense[j]
    b = dense[(j + 1) % m]
    sn = np.sin(th)
    w0 = np.where(th > 1e-12, np.sin((1.0 - f) * th) / np.where(sn > 1e-12, sn, 1.0), 1.0)
    w1 = np.where(th > 1e-12, np.sin(f * th) / np.where(sn > 1e-12, sn, 1.0), 0.0)
    out = w0[:, None] * a + w1[:, None] * b                 # slerp on each edge
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def _segseg_min_dists(P: np.ndarray) -> np.ndarray:
    """Minimum distances between every pair of NON-adjacent closed-polyline
    segments of P (n,4 in R⁴).  Returns the array of pairwise minima.

    Used as the topology gate for the random walk: a deformation that never
    lets two segments touch is an ambient isotopy, so the knot type cannot
    change.  Clamped closest-point-between-segments (Ericson, RTCD §5.1.9),
    vectorised over all candidate pairs.
    """
    n = len(P)
    A = P
    B = np.roll(P, -1, axis=0)          # segment i is A[i] -> B[i]
    d = B - A                           # (n,4) direction vectors

    # Candidate pairs (i,j), i<j, excluding adjacency on the closed loop.
    i, j = np.triu_indices(n, k=1)
    adj = (j == i + 1) | ((i == 0) & (j == n - 1))
    i, j = i[~adj], j[~adj]

    d1, d2 = d[i], d[j]
    r = A[i] - A[j]
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b

    s = np.where(denom > 1e-12, (b * f - c * e) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = (b * s + f) / np.where(e > 1e-12, e, 1.0)
    t = np.clip(t, 0.0, 1.0)
    s = np.clip((b * t - c) / np.where(a > 1e-12, a, 1.0), 0.0, 1.0)

    cp1 = A[i] + s[:, None] * d1
    cp2 = A[j] + t[:, None] * d2
    return np.linalg.norm(cp1 - cp2, axis=1)


def _seg_clearance_at(P: np.ndarray, k: int) -> float:
    """Min distance from the two segments incident to vertex k —
    (k-1,k) and (k,k+1) — to every non-adjacent segment.  O(n) gate for a
    single-vertex move."""
    n = len(P)
    A = P
    B = np.roll(P, -1, axis=0)
    best = np.inf
    for si in ((k - 1) % n, k % n):       # the two segments that moved
        d1 = B[si] - A[si]
        # all segments except si and its two loop-neighbours
        js = [j for j in range(n) if j not in (si, (si - 1) % n, (si + 1) % n)]
        Aj, Bj = A[js], B[js]
        d2 = Bj - Aj
        r = A[si] - Aj
        a = float(d1 @ d1)
        e = np.einsum("ij,ij->i", d2, d2)
        f = np.einsum("ij,ij->i", d2, r)
        c = d1 @ r.T
        b = d2 @ d1
        denom = a * e - b * b
        s = np.where(denom > 1e-12, (b * f - c * e) / np.where(denom > 1e-12, denom, 1.0), 0.0)
        s = np.clip(s, 0.0, 1.0)
        t = np.clip((b * s + f) / np.where(e > 1e-12, e, 1.0), 0.0, 1.0)
        s = np.clip((b * t - c) / a, 0.0, 1.0)
        cp1 = A[si] + s[:, None] * d1
        cp2 = Aj + t[:, None] * d2
        best = min(best, float(np.linalg.norm(cp1 - cp2, axis=1).min()))
    return best


def random_walk_knot(base_r3: np.ndarray, steps: int, amp: float,
                     seed: int, clearance: float = 0.04) -> np.ndarray:
    """Diffuse `base_r3` (n,3 in R³) far from its starting embedding via a
    topology-preserving random walk.

    The walk runs on the SAME geometry everything downstream uses — straight
    R³ segments of the stereographic image (what the .vect file stores and what
    knot_check evaluates).  Stereographic projection is a bijection
    R³ ↔ S³∖{N}, so any embedded R³ polyline is a valid S³ knot.

    Single-vertex Monte Carlo: each step jitters ONE random vertex (Gaussian
    scale `amp`).  The move is ACCEPTED only if the two incident segments keep
    clearance > `clearance` from all other strands.  No-tunnelling guarantee:
    every move is also capped at displacement < clearance/2, so the swept motion
    can never reach a strand that stays ≥ clearance away — no crossing is
    geometrically possible and the knot type is preserved exactly.
    """
    rng = np.random.default_rng(seed)
    pts = base_r3.copy()
    n = len(pts)
    step_cap = 0.5 * clearance
    accepted = 0
    for _ in range(steps):
        k = int(rng.integers(n))
        old = pts[k].copy()
        delta = rng.normal(scale=amp, size=3)
        if np.linalg.norm(delta) >= step_cap:         # too big a jump — could tunnel
            continue
        pts[k] = old + delta
        if _seg_clearance_at(pts, k) > clearance:
            accepted += 1
        else:
            pts[k] = old                              # revert
    drift = float(np.linalg.norm(pts - base_r3, axis=1).mean())
    print(f"  random walk: {accepted}/{steps} vertex moves accepted, "
          f"mean R³ drift {drift:.3f}", file=sys.stderr)
    return pts


def upsample_r3(pts: np.ndarray, n_target: int) -> np.ndarray:
    """Resample a closed R³ polyline to `n_target` vertices at EQUAL arc length.

    New vertices lie ON the existing segments (piecewise-linear interpolation),
    so the embedded curve is unchanged — same knot type, just finer.  This lets
    us random-walk cheaply at low N (large safe steps → far from the torus) and
    then densify for an accurate energy minimisation.
    """
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)   # closed edges
    cum = np.concatenate([[0.0], np.cumsum(seg)])                  # (n+1,)
    total = cum[-1]
    targets = np.linspace(0.0, total, n_target, endpoint=False)
    j = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(pts) - 1)
    f = np.where(seg[j] > 1e-12, (targets - cum[j]) / np.where(seg[j] > 1e-12, seg[j], 1.0), 0.0)
    a = pts[j]
    b = pts[(j + 1) % len(pts)]
    return a + f[:, None] * (b - a)


def s3_to_r3(pts: np.ndarray) -> np.ndarray:
    """
    Stereographic projection S³ → ℝ³ from north pole N = (0,0,0,1).

    Parameters
    ----------
    pts : (N, 4) array of unit vectors on S³

    Returns
    -------
    (N, 3) array in ℝ³
    """
    x4 = pts[:, 3]
    denom = 1.0 - x4
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError(
            "One or more points is at/near the north pole — "
            "projection is undefined. Perturb p, q, or n_points."
        )
    return pts[:, :3] / denom[:, np.newaxis]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_obj(pts_r3: np.ndarray, path: str, p: int, q: int) -> None:
    """Wavefront OBJ: vertices + closed polyline edges."""
    n = len(pts_r3)
    with open(path, "w") as f:
        f.write(f"# T({p},{q}) torus knot — generated by knots/generate.py\n")
        f.write(f"# {n} vertices\n\n")
        for x, y, z in pts_r3:
            f.write(f"v {x:.10f} {y:.10f} {z:.10f}\n")
        f.write("\n")
        for i in range(n):
            j = (i + 1) % n + 1   # 1-indexed next vertex; wraps n → 1
            f.write(f"l {i + 1} {j}\n")


def write_vect(pts_r3: np.ndarray, path: str) -> None:
    """
    .vect format — single-component closed curve:

        <n_components>
        <n_points>
        x y z
        ...
    """
    n = len(pts_r3)
    with open(path, "w") as f:
        f.write("1\n")
        f.write(f"{n}\n")
        for x, y, z in pts_r3:
            f.write(f"{x:.10f} {y:.10f} {z:.10f}\n")


def write_scene(pts_r3: np.ndarray, path: str, p: int, q: int,
                obj_name: str) -> None:
    """
    scene.txt for repulsive-curves (rcurves_app).

    The first line names the OBJ file (relative to scene.txt).
    Then repulsion parameters and constraints follow.
    """
    with open(path, "w") as f:
        f.write(f"# T({p},{q}) torus knot scene for rcurves_app\n")
        f.write(f"curve {obj_name}\n")
        f.write("repel_curve 3 6\n")      # tangent-point exponents α=3, β=6
        f.write("fix_barycenter\n")
        f.write(f"fix_totallength 2\n")   # normalise to length 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate T(p,q) torus knot → .obj, .vect, scene.txt"
    )
    parser.add_argument("p", type=int, help="First torus-knot parameter (p > 0)")
    parser.add_argument("q", type=int, help="Second torus-knot parameter (q > 0, gcd(p,q)=1)")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of sample points (default: 200)")
    parser.add_argument("--out", type=str, default=".",
                        help="Output directory (default: current directory)")
    parser.add_argument("--random-walk", action="store_true",
                        help="Diffuse the knot far from the Clifford torus via a "
                             "topology-preserving random walk (for convergence-basin tests)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for --random-walk (default: 0)")
    parser.add_argument("--walk-steps", type=int, default=120000,
                        help="Single-vertex moves to attempt (default: 120000)")
    parser.add_argument("--walk-amp", type=float, default=0.012,
                        help="Per-move Gaussian noise scale on S³ (default: 0.012)")
    parser.add_argument("--walk-clearance", type=float, default=0.05,
                        help="Min non-adjacent segment distance kept during walk (default: 0.05)")
    parser.add_argument("--upsample-to", type=int, default=None,
                        help="After the walk, resample to this many points at equal arc "
                             "length (same knot, finer). Walk cheaply at low --n, densify here.")
    args = parser.parse_args()

    if args.p <= 0 or args.q <= 0:
        parser.error("p and q must be positive integers")
    g = math.gcd(args.p, args.q)
    if g != 1:
        print(
            f"Warning: gcd({args.p},{args.q}) = {g} ≠ 1 "
            "— this is a link, not a knot.",
            file=sys.stderr,
        )

    os.makedirs(args.out, exist_ok=True)
    prefix = f"T{args.p}_{args.q}"

    s3_pts = torus_knot_s3(args.p, args.q, args.n)
    r3_pts = s3_to_r3(s3_pts)
    if args.random_walk:
        # Walk in R³ on the straight segments knot_check / the .vect actually use.
        r3_pts = random_walk_knot(r3_pts, args.walk_steps, args.walk_amp,
                                  args.seed, args.walk_clearance)
    if args.upsample_to:
        r3_pts = upsample_r3(r3_pts, args.upsample_to)
        print(f"  upsampled {len(s3_pts)} → {args.upsample_to} points at equal arc length",
              file=sys.stderr)

    obj_path   = os.path.join(args.out, f"{prefix}.obj")
    vect_path  = os.path.join(args.out, f"{prefix}.vect")
    scene_path = os.path.join(args.out, "scene.txt")

    write_obj(r3_pts, obj_path, args.p, args.q)
    write_vect(r3_pts, vect_path)
    # scene.txt references the OBJ by filename only (same directory)
    write_scene(r3_pts, scene_path, args.p, args.q, f"{prefix}.obj")

    print(f"T({args.p},{args.q}) knot  —  {len(r3_pts)} points")
    print(f"  OBJ:   {obj_path}")
    print(f"  VECT:  {vect_path}")
    print(f"  Scene: {scene_path}")


if __name__ == "__main__":
    main()
