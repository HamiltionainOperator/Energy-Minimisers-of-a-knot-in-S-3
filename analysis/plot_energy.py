#!/usr/bin/env python3
"""
Plot energy_log.csv produced by Repulsor/energy_s3.

Expected CSV columns (any superset of):
    iteration, energy [, gradient_norm] [, step_size] [, ...]

The first column whose name contains 'iter' is used as the x-axis.
The first column whose name contains 'energy' is plotted on the primary panel.
Any remaining columns are plotted on additional sub-panels.

Usage:
    python3 analysis/plot_energy.py output/energy_log.csv
    python3 analysis/plot_energy.py output/energy_log.csv --out figs/energy.png
    python3 analysis/plot_energy.py output/energy_log.csv --title "T(3,5) on S³"

Dependencies: matplotlib (and optionally pandas)
    pip3 install matplotlib pandas
"""

import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# CSV reader (no pandas required)
# ---------------------------------------------------------------------------

def load_csv(path: str) -> Dict[str, np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return cols


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(cols: Dict[str, np.ndarray], out_path: str, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("Error: matplotlib not installed.  Run: pip3 install matplotlib",
              file=sys.stderr)
        sys.exit(1)

    keys = list(cols)

    # Identify axes
    iter_col   = next((k for k in keys if "iter" in k.lower()), keys[0])
    energy_col = next((k for k in keys if "energy" in k.lower()),
                      keys[1] if len(keys) > 1 else keys[0])
    extra_cols = [k for k in keys if k not in (iter_col, energy_col)]

    iters  = cols[iter_col]
    energy = cols[energy_col]
    n_panels = 1 + len(extra_cols)

    fig = plt.figure(figsize=(10, 3 * n_panels), constrained_layout=True)
    gs  = gridspec.GridSpec(n_panels, 1, figure=fig)

    # ── Energy panel ────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.semilogy(iters, energy, color="#2563EB", linewidth=1.6,
                 label="Tangent-point energy")
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Energy  (log scale)")
    ax0.set_title(title)
    ax0.grid(True, which="both", alpha=0.25)
    ax0.legend(fontsize=9)

    # Annotate start / end
    ax0.annotate(f"{energy[0]:.3g}", xy=(iters[0], energy[0]),
                 xytext=(5, 5), textcoords="offset points", fontsize=8, color="gray")
    ax0.annotate(f"{energy[-1]:.3g}", xy=(iters[-1], energy[-1]),
                 xytext=(-45, 5), textcoords="offset points", fontsize=8, color="gray")

    # ── Extra panels ─────────────────────────────────────────────────────────
    _colors = ["#DC2626", "#16A34A", "#9333EA", "#EA580C"]
    for k, col in enumerate(extra_cols):
        ax = fig.add_subplot(gs[k + 1])
        vals  = cols[col]
        color = _colors[k % len(_colors)]

        if any(kw in col.lower() for kw in ("grad", "norm", "residual")):
            ax.semilogy(iters, np.abs(vals), color=color, linewidth=1.2)
        elif any(kw in col.lower() for kw in ("step", "size", "alpha")):
            ax.semilogy(iters, np.abs(vals), color=color, linewidth=1.2)
        else:
            ax.plot(iters, vals, color=color, linewidth=1.2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel(col)
        ax.grid(True, which="both", alpha=0.25)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  →  {out_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(cols: Dict[str, np.ndarray]) -> None:
    keys       = list(cols)
    energy_col = next((k for k in keys if "energy" in k.lower()), keys[0])
    e          = cols[energy_col]

    print(f"\n{'─'*40}")
    print(f"  Energy column : {energy_col}")
    print(f"  Iterations    : {len(e)}")
    print(f"  Initial value : {e[0]:.8g}")
    print(f"  Final value   : {e[-1]:.8g}")
    print(f"  Minimum       : {e.min():.8g}  @ iter {e.argmin()}")
    print(f"  Reduction     : {e[0]/e[-1]:.4f}× ({100*(1-e[-1]/e[0]):.1f}% decrease)")
    print(f"{'─'*40}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot energy_log.csv from the S³ knot energy pipeline"
    )
    parser.add_argument("csv",   help="Input CSV file (energy_log.csv)")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: <csv>.png)")
    parser.add_argument("--title", default="Knot Energy on S³",
                        help="Plot title")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or os.path.splitext(args.csv)[0] + ".png"
    cols     = load_csv(args.csv)
    print_summary(cols)
    plot(cols, out_path, title=args.title)


if __name__ == "__main__":
    main()
