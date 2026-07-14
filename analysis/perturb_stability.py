#!/usr/bin/env python3
"""
Perturbation-stability test for a minimum of the S³ O'Hara energy.

Idea (finite-perturbation / numerical-Hessian probe of a local minimum):
take a converged minimum M* with energy E*, kick every vertex by a small
random displacement, then re-run the SAME minimiser.  A genuine local minimum
must

    (1) never re-minimise to an energy BELOW E*          (else M* was a saddle)
    (2) return to E* within tolerance                    (energy is the
        symmetry-invariant scalar: reparam / rigid / Möbius all preserve it)
    (3) return to the same SHAPE                          (rules out "same
        energy, different basin")

and the knot type must stay det 3 throughout (the kick must not retie it).

This script runs ONE (amp_frac, seed) condition end-to-end and prints a single
JSON line so a driver / workflow can fan many conditions out and aggregate.

Usage
-----
    python3 analysis/perturb_stability.py \
        --ref output/T2_3_ref/T2_3_s3.vect --estar 54.25878099 \
        --binary build/energy_s3 --amp-frac 0.25 --seed 1 \
        --iter 3000 --step 0.01 --outdir output/_perturb/a0.25_s1
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from knot_check import read_vect_pts, robust_determinant   # noqa: E402


# ---------------------------------------------------------------------------
# .vect writer (single closed component) — matches knots/generate.py
# ---------------------------------------------------------------------------

def write_vect(pts: np.ndarray, path: str) -> None:
    n = len(pts)
    with open(path, "w") as f:
        f.write("1\n")
        f.write(f"{n}\n")
        for x, y, z in pts:
            f.write(f"{x:.10f} {y:.10f} {z:.10f}\n")


# ---------------------------------------------------------------------------
# Invariant shape descriptor
# ---------------------------------------------------------------------------

def shape_fingerprint(pts: np.ndarray) -> np.ndarray:
    """Sorted, scale-normalised multiset of all pairwise distances.

    Invariant to rigid motion, reflection AND reparametrisation (it is an
    unordered set of distances), and to scale (normalised by total curve
    length).  Two curves with near-zero fingerprint distance have the same
    shape up to those symmetries.
    """
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    L = seg.sum()
    d = pts / L                                   # scale-normalise
    diff = d[:, None, :] - d[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(len(pts), k=1)
    return np.sort(dist[iu])


def shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMS difference of two equal-length sorted fingerprints."""
    fa, fb = shape_fingerprint(a), shape_fingerprint(b)
    return float(np.sqrt(np.mean((fa - fb) ** 2)))


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------

def run_condition(ref_path, estar, binary, amp_frac, seed, iters, step, outdir):
    os.makedirs(outdir, exist_ok=True)
    ref = read_vect_pts(ref_path)
    seg = np.linalg.norm(np.roll(ref, -1, axis=0) - ref, axis=1)
    mean_edge = float(seg.mean())

    # 1. Perturb: Gaussian kick, std = amp_frac · mean edge length.
    rng = np.random.default_rng(seed)
    amp = amp_frac * mean_edge
    perturbed = ref + rng.normal(scale=amp, size=ref.shape)
    pert_path = os.path.join(outdir, "perturbed.vect")
    write_vect(perturbed, pert_path)
    pert_disp = float(np.linalg.norm(perturbed - ref, axis=1).mean())

    # 2. Topology of the perturbed (pre-minimisation) curve.
    det_pert, _, _ = robust_determinant(perturbed, ntries=11)

    # 3. Re-minimise from the perturbed state with the SAME minimiser.
    log_path = os.path.join(outdir, "energy_log.csv")
    out_path = os.path.join(outdir, "remin_s3.vect")
    proc = subprocess.run(
        [binary, pert_path, log_path, out_path, str(iters), str(step)],
        capture_output=True, text=True,
    )
    txt = proc.stdout + proc.stderr

    def _grab(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None

    e_perturbed = _grab(r"Initial E\^\(2\)_\{S³\} energy:\s*([0-9.eE+-]+)")
    e_final = _grab(r"Final\s+E\^\(2\)_\{S³\} energy\s*:\s*([0-9.eE+-]+)")

    gnorm_final = None
    if os.path.exists(log_path):
        with open(log_path) as f:
            rows = [r for r in f.read().splitlines() if r.strip()]
        if rows:
            last = rows[-1].split(",")
            if len(last) >= 3:
                e_final = float(last[1])          # authoritative from log
                gnorm_final = float(last[2])

    # 4. Shape + topology of the re-minimised result vs the reference minimum.
    shape_d, det_final = None, None
    if os.path.exists(out_path):
        remin = read_vect_pts(out_path)
        shape_d = shape_distance(ref, remin)
        det_final, _, _ = robust_determinant(remin, ntries=11)

    dE = (e_final - estar) if e_final is not None else None
    result = {
        "amp_frac": amp_frac,
        "seed": seed,
        "mean_edge": round(mean_edge, 6),
        "perturb_disp": round(pert_disp, 6),
        "det_perturbed": det_pert,
        "det_final": det_final,
        "E_perturbed": e_perturbed,
        "E_final": e_final,
        "Estar": estar,
        "dE": dE,                      # E_final - E*  (must be ≳ 0, ≈ 0)
        "went_below": (dE is not None and dE < -1e-3),
        "gnorm_final": gnorm_final,
        "shape_dist": shape_d,
        "topology_ok": (det_pert == 3 and det_final == 3),
        "rc": proc.returncode,
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="One perturbation-stability condition")
    ap.add_argument("--ref", required=True, help="reference minimum .vect (M*)")
    ap.add_argument("--estar", type=float, required=True, help="reference energy E*")
    ap.add_argument("--binary", default="build/energy_s3")
    ap.add_argument("--amp-frac", type=float, required=True,
                    help="kick std as a fraction of mean edge length")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iter", type=int, default=3000)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    res = run_condition(args.ref, args.estar, args.binary, args.amp_frac,
                        args.seed, args.iter, args.step, args.outdir)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
