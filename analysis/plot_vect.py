#!/usr/bin/env python3
"""
Render a .vect knot in 3-D with two distribution diagnostics overlaid:

  • vertex dots   — a subsample of the vertices (every n/200-th). Even spacing
                    along the curve ⇒ the dots look evenly strewn; clustering
                    ⇒ they visibly bunch on one lobe and thin out elsewhere.
  • tangent arrows — ~200 forward-difference arrows, drawn UN-normalised so each
                    arrow's LENGTH is the local edge length. A well-parametrised
                    curve gives arrows of near-equal length that turn smoothly;
                    reparametrisation failure shows up as arrows that suddenly
                    jump in direction or swing wildly in length.

Both are subsampled to ~200 markers so an N=10000 knot stays readable (all
10000 dots/arrows would be an unreadable blob).

Usage: plot_vect.py <input.vect> <output.png> [--arrows K]
"""
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt


def read_vect(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    # Format:  1 (components) / N (points) / "x y z" * N
    n_pts = int(lines[1])
    pts = [[float(x) for x in line.split()] for line in lines[2 : 2 + n_pts]]
    return np.array(pts)


def main():
    ap = argparse.ArgumentParser(description="Render a .vect knot with distribution diagnostics")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--arrows", type=int, default=200,
                    help="approx. number of tangent arrows / vertex dots to draw (default 200)")
    args = ap.parse_args()

    pts = read_vect(args.input)
    n = len(pts)

    # ── point-distribution stats (in the rendered ℝ³ image) ────────────────────
    edges = np.roll(pts, -1, axis=0) - pts          # closed-loop edge vectors
    elen = np.linalg.norm(edges, axis=1)
    ratio = elen.max() / max(elen.min(), 1e-12)
    print(f"{args.input}: {n} vertices")
    print(f"  R³ edge length  min={elen.min():.4g}  max={elen.max():.4g}  "
          f"mean={elen.mean():.4g}  max/min={ratio:.1f}")
    if ratio > 5:
        print("  ⚠ strong spacing non-uniformity in the ℝ³ image "
              "(may be a stereographic-projection artifact; the energy lives on S³)")

    # ── subsample for legible markers (~args.arrows of each) ───────────────────
    step = max(1, n // max(1, args.arrows))
    idx = np.arange(0, n, step)
    sp = pts[idx]
    st = edges[idx]                                  # local tangent (edge) vectors

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # full curve
    pts_closed = np.vstack([pts, pts[0]])
    ax.plot(pts_closed[:, 0], pts_closed[:, 1], pts_closed[:, 2],
            "-", linewidth=1.5, color="#1f77b4", alpha=0.7)

    # vertex dots (subsampled) — clustering shows as bunching
    ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], c="red", s=10, depthshade=True)

    # equal aspect
    rng = (pts.max(0) - pts.min(0)).max() / 2.0

    # tangent arrows (subsampled). Drawn with a COMMON scale factor so each
    # arrow's length stays proportional to its local edge length — uniform
    # spacing ⇒ equal-length arrows; clustering ⇒ short arrows in dense regions,
    # long ones in sparse. The scale just makes the median arrow ~8% of the box
    # so they're visible (raw edges are ~1e-3 of the curve span at N=10000).
    med = np.median(np.linalg.norm(st, axis=1))
    qscale = (0.08 * 2 * rng) / max(med, 1e-12)
    ax.quiver(sp[:, 0], sp[:, 1], sp[:, 2],
              st[:, 0], st[:, 1], st[:, 2],
              length=qscale, normalize=False, color="#2ca02c", linewidth=1.2,
              arrow_length_ratio=0.35)

    mid = (pts.max(0) + pts.min(0)) * 0.5
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[1] - rng, mid[1] + rng)
    ax.set_zlim(mid[2] - rng, mid[2] + rng)
    ax.set_axis_off()
    ax.set_title(f"{n} vtx · {len(idx)} dots+arrows · edge max/min {ratio:.1f}",
                 color="#444", fontsize=10)

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"  → saved 3D render with diagnostics to {args.output}")


if __name__ == "__main__":
    main()
