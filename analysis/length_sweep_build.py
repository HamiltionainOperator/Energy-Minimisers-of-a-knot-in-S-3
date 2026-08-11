#!/usr/bin/env python3
"""
Build a one-parameter family of curves realising ONE fixed knot type (the
trefoil, taken from the pipeline's converged E_q minimiser) whose S³ length
sweeps 0 → ∞.

Three segments, glued into a single monotone L axis:

  S  (shrink)   : ℝ³ similarity  y ↦ s·y,  s: 1 → 1e-5.
                  Pure ℝ³ isotopy ⇒ knot type fixed BY CONSTRUCTION.
                  Curve collapses into a small ball ⇒ S³ length → 0.
  U  (scale up) : ℝ³ similarity  s: 1 → 1e3  (secondary branch; shows that
                  scaling ALONE cannot reach L→∞ — the image is trapped in a
                  spherical cap and L turns around and falls back to 0).
  G  (grow)     : along-strand helix in a CLOSED rotation-minimising frame,
                  G1 = amplitude ramp A: 0 → A*, G2 = frequency ramp k: k0 → kmax.
                  Also an ℝ³ isotopy as long as the tube stays embedded;
                  certified a posteriori with the determinant oracle.

Writes one .vect per sample plus a manifest; energies come from the pipeline
binary (`build/energy_s3 --eval`).
"""
import json
import os
import sys

import numpy as np

ROOT = "/Users/yash/knot-s3"
OUT = os.environ.get("LSWEEP_DIR", os.path.join(ROOT, "output/Lsweep"))
BASE = os.environ.get("LSWEEP_BASE",
                      os.path.join(ROOT, "output/T2_3_q/T2_3_s3.vect"))

A_STAR = float(os.environ.get("LSWEEP_A", 0.18))   # helix amplitude (ℝ³); base curve has min strand gap 0.478
                  # and 1/κ_max = 1.06, so 2A = 0.36 keeps the tube embedded
K0 = 16           # frequency at the end of the amplitude ramp
PTS_PER_PITCH = 10  # vertices per inter-turn spacing (discretisation rule)
PPP_LONG = 4      # coarser rule for the long tail (3e-4 error, see below)
K_LONG_MAX = int(os.environ.get("LSWEEP_KMAX", 2000))


def read_vect(path):
    tok = open(path).read().split()
    n = int(tok[1])
    return np.array(tok[2:2 + 3 * n], float).reshape(n, 3)


def write_vect(pts, path):
    with open(path, "w") as f:
        f.write("1\n%d\n" % len(pts))
        for x, y, z in pts:
            f.write("%.17g %.17g %.17g\n" % (x, y, z))


def resample_periodic(pts, m):
    """Spectral (band-limited) periodic upsample, parameterised by the existing
    index — the optimiser leaves the curve uniform in S³ arc length, so index ≈
    S³ arc length, which is the parameterisation we want to keep.  Exact for a
    smooth closed curve and free of the spline's endpoint fuss."""
    n = len(pts)
    if m == n:
        return pts.copy()
    F = np.fft.rfft(pts, axis=0)
    G = np.zeros((m // 2 + 1, 3), complex)
    kk = min(len(F), len(G))
    G[:kk] = F[:kk]
    if n % 2 == 0 and kk == len(F) and len(G) > len(F) - 1:
        G[len(F) - 1] *= 0.5          # split the Nyquist bin
    return np.fft.irfft(G, m, axis=0) * (m / n)


def closed_rmf(pts):
    """Rotation-minimising frame (double reflection) with the holonomy removed
    by a uniform counter-twist, so the frame closes up around the loop.
    Without this the helix has a seam kink that dominates the energy."""
    n = len(pts)
    T = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)

    # seed
    a = np.array([0.0, 0.0, 1.0])
    if abs(T[0] @ a) > 0.9:
        a = np.array([1.0, 0.0, 0.0])
    u0 = a - (a @ T[0]) * T[0]
    u0 /= np.linalg.norm(u0)

    U = np.zeros_like(pts)
    U[0] = u0
    u = u0
    for i in range(1, n + 1):
        t0, t1 = T[i - 1], T[i % n]
        # double-reflection transport of u from t0 to t1
        v1 = pts[i % n] - pts[i - 1]
        c1 = v1 @ v1
        if c1 > 0:
            uL = u - (2.0 / c1) * (v1 @ u) * v1
            tL = t0 - (2.0 / c1) * (v1 @ t0) * v1
            v2 = t1 - tL
            c2 = v2 @ v2
            u = uL - (2.0 / c2) * (v2 @ uL) * v2 if c2 > 1e-30 else uL
        u = u - (u @ t1) * t1
        u /= np.linalg.norm(u)
        if i < n:
            U[i] = u

    # holonomy of the transported frame after one full loop
    uf = u
    c = np.clip(uf @ U[0], -1, 1)
    V0 = np.cross(T[0], U[0])
    alpha = np.arctan2(V0 @ uf, c)          # signed defect angle

    # uniform counter-twist so the frame closes
    beta = -alpha * np.arange(n) / n
    V = np.cross(T, U)
    Uc = np.cos(beta)[:, None] * U + np.sin(beta)[:, None] * V
    Vc = np.cross(T, Uc)
    return Uc, Vc, alpha


def helix_res(base_len, A, k):
    """Vertex count: the tightest length scale in the coil is the inter-turn
    spacing (pitch = L₀/k), so resolve THAT, not the turn itself."""
    if k == 0 or A == 0.0:
        return 1000
    pitch = base_len / k
    turn = np.hypot(pitch, 2 * np.pi * A)
    return int(np.clip(PTS_PER_PITCH * turn * k / pitch, 1000, 60000))


def helix(base, A, k, m):
    """base curve upsampled to m points + helix of amplitude A, k turns."""
    P = resample_periodic(base, m)
    U, V, alpha = closed_rmf(P)
    th = 2 * np.pi * k * np.arange(m) / m
    return P + A * (np.cos(th)[:, None] * U + np.sin(th)[:, None] * V)


def min_gap(P, skip_frac=0.02, chunk=2048, n_max=70000):
    """Closest approach between curve points more than skip_frac·n apart in
    index (chunked — n can be 6e4 here, so no dense n×n matrix).  Skipped
    outright on the long tail: it is an O(n²) numpy scan, and there the
    embedded-tube argument (A < reach of the base curve) is the certificate,
    not this diagnostic."""
    n = len(P)
    if n > n_max:
        return float("nan")
    skip = max(3, int(skip_frac * n))
    idx = np.arange(n)
    best = np.inf
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        D = np.linalg.norm(P[a:b, None, :] - P[None, :, :], axis=-1)
        sep = np.abs(idx[a:b, None] - idx[None, :])
        sep = np.minimum(sep, n - sep)
        D[sep < skip] = np.inf
        best = min(best, float(D.min()))
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    base = read_vect(BASE)
    man = []

    def add(tag, pts, seg, param):
        p = os.path.join(OUT, tag + ".vect")
        write_vect(pts, p)
        d = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)
        man.append(dict(tag=tag, path=p, seg=seg, param=param, n=len(pts),
                        L_r3=float(d.sum()), gap=min_gap(pts),
                        edge=float(d.mean())))
        print("  %-18s n=%5d  L_R3=%9.3f  gap=%.4f" %
              (tag, len(pts), d.sum(), man[-1]["gap"]))

    # ── segment S: shrink (and the s=1 anchor = the pipeline minimiser) ──
    # 1e-5 is the floor: the energy uses θ = arccos(x·y), which loses all
    # precision once 1−x·y drops to machine epsilon (θ ≳ 2e-8 rad).  At s=1e-5
    # the nearest-neighbour θ is already ~4e-7 and E★ scatters by ~0.1 %.
    print("[S] shrink branch  s: 1 → 1e-5")
    for s in np.logspace(0, -5, 41):
        add("S_s%.3e" % s, base * s, "shrink", float(s))

    # ── segment U: scale up (secondary; L caps out) ──
    print("[U] scale-up branch  s: 1 → 1e3")
    for s in np.logspace(0, 3, 25)[1:]:
        add("U_s%.3e" % s, base * s, "scaleup", float(s))

    # ── segment G1: amplitude ramp at fixed k ──
    print("[G1] amplitude ramp  A: 0 → %.3f at k=%d" % (A_STAR, K0))
    L0 = float(np.linalg.norm(np.roll(base, -1, 0) - base, axis=1).sum())
    m1 = helix_res(L0, A_STAR, K0)
    for A in np.linspace(0.0, A_STAR, 9):
        add("G1_A%.4f" % A, helix(base, A, K0, m1), "ampramp", float(A))

    # ── segment G2: frequency ramp at fixed amplitude ──
    print("[G2] frequency ramp  k: %d → 320 at A=%.3f" % (K0, A_STAR))
    ks = [16, 20, 26, 32, 40, 52, 64, 80, 104, 128, 160, 200, 256, 320]
    for k in ks:
        m = helix_res(L0, A_STAR, k)
        add("G2_k%04d" % k, helix(base, A_STAR, k, m), "freqramp", float(k))

    # ── segment V: dense sampling through the valley, where the scaling fold
    #    (L is MAXIMAL at s=1) hands over to the coil branch ──
    print("[V] valley refinement")
    seen = {r["tag"] for r in man}
    for s in np.linspace(0.40, 1.60, 25):
        tag = "V_s%.4f" % s
        if tag not in seen:
            add(tag, base * s, "shrink" if s <= 1 else "scaleup", float(s))
    for A in np.linspace(0.0025, 0.0225, 9):
        tag = "V_A%.4f" % A
        if tag not in seen:
            add(tag, helix(base, A, K0, 1000), "ampramp", float(A))

    # ── segment G3: the long tail, L → O(10³) ──────────────────────────────
    # The coil packs its length onto the cylindrical SURFACE of the tube, so
    # strand spacing ~ (tube area)/L and the vertex count grows like k².  A
    # resolution study at k=320 (PTS_PER_PITCH = 3…15) puts the discretisation
    # error at 3e-4 already for 3 points per pitch, so the tail uses 4.
    if "--long" in sys.argv:
        print("[G3] long coil  k: 400 → %d at %d pts/pitch" % (K_LONG_MAX, PPP_LONG))
        for k in [400, 512, 640, 800, 1000, 1280, 1600, 2000, 2560, 3200]:
            if k > K_LONG_MAX:
                break
            pitch = L0 / k
            turn = np.hypot(pitch, 2 * np.pi * A_STAR)
            m = int(PPP_LONG * turn * k / pitch)
            add("G3_k%05d" % k, helix(base, A_STAR, k, m), "freqramp", float(k))

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("\n%d curves → %s" % (len(man), OUT))


if __name__ == "__main__":
    main()
