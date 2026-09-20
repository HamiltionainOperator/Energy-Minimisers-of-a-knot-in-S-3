#!/usr/bin/env python3
"""
Numerical test for the proposed long-strand decomposition of E_{S3}/L^2.

The construction is deliberately explicit and diagnostic-heavy:

  * a compact core tangle lives in a prescribed R3 ball before stereographic
    lift to S3;
  * the same long strand is used for the trefoil-core and unknot-core curves;
  * vertices are labelled core/strand, so the O'Hara quadrature is split into
    core-core, strand-strand, and cross terms;
  * junction angles and core-radius drift are reported, because a bad stitch
    can dominate the experiment.

The discrete energy matches Repulsor/energy_s3.cpp: cyclic non-adjacent vertex
pairs, geodesic S3 ambient distance, shorter curve arclength distance, Voronoi
weights, and a final factor of 2 for the ordered double integral.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(ROOT, "output", "long_strand_decomp")


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-300)


def r3_to_s3(pts: np.ndarray) -> np.ndarray:
    r2 = np.sum(pts * pts, axis=-1, keepdims=True)
    return np.concatenate([2.0 * pts / (r2 + 1.0), (r2 - 1.0) / (r2 + 1.0)], axis=-1)


def s3_to_r3(pts: np.ndarray) -> np.ndarray:
    d = 1.0 - pts[:, 3:4]
    return pts[:, :3] / np.maximum(d, 1e-300)


def write_vect(path: str, pts: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(f"1\n{len(pts)}\n")
        for p in pts:
            f.write("%.17g %.17g %.17g\n" % tuple(p))


def write_s4(path: str, pts: np.ndarray, labels: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(f"{len(pts)}\n")
        for p, lab in zip(pts, labels):
            f.write("%.17g %.17g %.17g %.17g %d\n" % (p[0], p[1], p[2], p[3], int(lab)))


def geodesic_edges(s4: np.ndarray) -> np.ndarray:
    dots = np.einsum("ij,ij->i", s4, np.roll(s4, -1, axis=0))
    return np.arccos(np.clip(dots, -1.0, 1.0))


def slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    c = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    th = np.arccos(c)
    st = np.sin(th)
    safe = np.where(st > 1e-14, st, 1.0)
    w0 = np.where(th > 1e-14, np.sin((1.0 - t) * th) / safe, 1.0 - t)
    w1 = np.where(th > 1e-14, np.sin(t * th) / safe, t)
    out = w0[:, None] * a + w1[:, None] * b
    return normalize(out)


def resample_closed_s3(s4: np.ndarray, labels: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = geodesic_edges(s4)
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    total = float(cum[-1])
    targets = np.linspace(0.0, total, n, endpoint=False)
    j = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(s4) - 1)
    frac = np.where(edges[j] > 1e-14, (targets - cum[j]) / np.maximum(edges[j], 1e-300), 0.0)
    out = slerp(s4[j], s4[(j + 1) % len(s4)], frac)
    return out, labels[j].copy()


def resample_closed_s3_with_passes(s4: np.ndarray, labels: np.ndarray, pass_ids: np.ndarray,
                                   n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample while retaining the contiguous strand-pass ownership labels."""
    edges = geodesic_edges(s4)
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    total = float(cum[-1])
    targets = np.linspace(0.0, total, n, endpoint=False)
    j = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(s4) - 1)
    frac = np.where(edges[j] > 1e-14, (targets - cum[j]) / np.maximum(edges[j], 1e-300), 0.0)
    out = slerp(s4[j], s4[(j + 1) % len(s4)], frac)
    return out, labels[j].copy(), pass_ids[j].copy()


def bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int, endpoint: bool) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, endpoint=endpoint)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def trefoil_arc_core(eps: float, n: int) -> np.ndarray:
    """A compact open trefoil tangle from A=(-a,0,0) to B=(a,0,0)."""
    a = 0.38 * eps
    t = np.linspace(0.08 * math.pi, 1.72 * math.pi, n)
    raw = np.column_stack([
        (2.0 + np.cos(3 * t)) * np.cos(2 * t),
        (2.0 + np.cos(3 * t)) * np.sin(2 * t),
        np.sin(3 * t),
    ])
    raw -= 0.5 * (raw[0] + raw[-1])
    e = raw[-1] - raw[0]
    ehat = e / np.linalg.norm(e)
    target = np.array([1.0, 0.0, 0.0])
    v = np.cross(ehat, target)
    s = np.linalg.norm(v)
    c = float(ehat @ target)
    if s < 1e-14:
        if c > 0.0:
            R = np.eye(3)
        else:
            R = np.diag([-1.0, -1.0, 1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
    q = np.einsum("ij,kj->ik", raw, R)
    q[:, 0] *= (2.0 * a) / (q[-1, 0] - q[0, 0])
    q[:, 1:] *= (0.42 * eps) / max(np.max(np.linalg.norm(q[:, 1:], axis=1)), 1e-300)
    q[:, 0] -= 0.5 * (q[0, 0] + q[-1, 0])
    return q


def unknot_arc_core(eps: float, n: int) -> np.ndarray:
    a = 0.38 * eps
    t = np.linspace(0.0, 1.0, n)
    x = -a + 2.0 * a * t
    y = 0.16 * eps * np.sin(math.pi * t)
    z = 0.04 * eps * np.sin(2.0 * math.pi * t)
    return np.column_stack([x, y, z])


def smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def long_strand(eps: float, far: float, lateral_sep: float, n_arm: int, n_cap: int) -> np.ndarray:
    """Shared non-winding out-and-back strand from B to A.

    In stereographic R3 this is two nearly straight radial arms plus one far
    cap.  After inverse stereographic lift, the arms are close to one great
    circle passing through the north pole p=(0,0,0,1): the strand goes from the
    core toward +infinity, crosses near p, and returns from -infinity.  The cap
    only keeps the finite-R3 representation embedded; it is not a helix and
    does not add turns.
    """
    a = 0.38 * eps
    b = np.array([a, 0.0, 0.0])
    c = np.array([-a, 0.0, 0.0])
    t = np.linspace(0.0, 1.0, n_arm, endpoint=False)
    h = smoothstep(t)
    out = np.column_stack([
        b[0] + (far - b[0]) * t,
        lateral_sep * h,
        np.zeros_like(t),
    ])
    th = np.linspace(0.0, math.pi, n_cap, endpoint=False)
    cap = np.column_stack([
        far * np.cos(th),
        lateral_sep * np.cos(th),
        far * np.sin(th),
    ])
    t2 = np.linspace(0.0, 1.0, n_arm, endpoint=True)
    h2 = smoothstep(t2)
    inn = np.column_stack([
        -far + (c[0] + far) * t2,
        -lateral_sep * (1.0 - h2),
        np.zeros_like(t2),
    ])
    return np.vstack([out, cap, inn])


def raw_curve(kind: str, eps: float, far: float, lateral_sep: float, n_arm: int, n_cap: int) -> Tuple[np.ndarray, np.ndarray]:
    n_core = 900
    core = trefoil_arc_core(eps, n_core) if kind == "trefoil" else unknot_arc_core(eps, n_core)
    strand = long_strand(eps, far, lateral_sep, n_arm, n_cap)
    pts = np.vstack([core[:-1], strand[:-1]])
    labels = np.concatenate([np.zeros(len(core) - 1, dtype=np.int8), np.ones(len(strand) - 1, dtype=np.int8)])
    return pts, labels


def curve_for_length(kind: str, target_L: float, eps: float, lateral_sep: float, min_far: float, max_far: float, n_per_len: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    n_arm = 1200
    n_cap = 240

    def length_at(far: float) -> float:
        pts, _ = raw_curve(kind, eps, far, lateral_sep, n_arm, n_cap)
        return float(geodesic_edges(r3_to_s3(pts)).sum())

    L_min = length_at(min_far)
    L_max = length_at(max_far)
    reached = L_min <= target_L <= L_max
    if target_L <= L_min:
        far = min_far
    elif target_L >= L_max:
        far = max_far
    else:
        lo, hi = min_far, max_far
        for _ in range(48):
            mid = math.sqrt(lo * hi)
            L = length_at(mid)
            if L < target_L:
                lo = mid
            else:
                hi = mid
        far = hi

    pts, labels = raw_curve(kind, eps, far, lateral_sep, n_arm, n_cap)
    s4 = r3_to_s3(pts)
    raw_L = float(geodesic_edges(s4).sum())
    n = int(max(400, min(24000, math.ceil(n_per_len * raw_L))))
    s4, labels = resample_closed_s3(s4, labels, n)
    L = float(geodesic_edges(s4).sum())
    info = {
        "strand_model": "geodesic_out_and_back",
        "target_reached": float(reached),
        "min_model_length": L_min,
        "max_model_length": L_max,
        "far": far,
        "lateral_sep": lateral_sep,
        "n": float(n),
        "length": L,
    }
    return s4, labels, info


def curve_for_far(kind: str, far: float, eps: float, lateral_sep: float, n_per_len: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    n_arm = 1200
    n_cap = 240
    pts, labels = raw_curve(kind, eps, far, lateral_sep, n_arm, n_cap)
    s4 = r3_to_s3(pts)
    raw_L = float(geodesic_edges(s4).sum())
    n = int(max(400, min(24000, math.ceil(n_per_len * raw_L))))
    s4, labels = resample_closed_s3(s4, labels, n)
    L = float(geodesic_edges(s4).sum())
    return s4, labels, {
        "strand_model": "geodesic_out_and_back",
        "target_reached": 1.0,
        "min_model_length": float("nan"),
        "max_model_length": float("nan"),
        "far": far,
        "lateral_sep": lateral_sep,
        "n": float(n),
        "length": L,
    }


def meridian(s: np.ndarray, phi: float) -> np.ndarray:
    """A great-circle meridian from the stereographic south pole to north."""
    direction = np.array([math.cos(phi), math.sin(phi), 0.0])
    return np.column_stack([np.sin(s)[:, None] * direction[None, :], -np.cos(s)])


def latitude_cap(s: float, phi0: float, phi1: float, n: int) -> np.ndarray:
    phis = np.linspace(phi0, phi1, n, endpoint=False)
    return np.column_stack([
        np.sin(s) * np.cos(phis), np.sin(s) * np.sin(phis), np.zeros(n), -np.cos(s) * np.ones(n)
    ])


def multi_return_curve(kind: str, pass_count: int, eps: float, lane_radius: float,
                       n_per_len: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Core plus an embedded, controlled bundle of near-great-circle passes.

    ``pass_count`` is even.  Pass j follows the meridian at azimuth
    phi_j=j*pi/(k-1), alternately south-to-north and north-to-south.  Adjacent
    passes are connected on fixed-radius latitude circles around the two poles.
    The bundle is therefore a single closed strand, with no helical turns.
    """
    if pass_count < 2 or pass_count % 2:
        raise ValueError("pass_count must be an even integer >= 2")
    core = trefoil_arc_core(eps, 900) if kind == "trefoil" else unknot_arc_core(eps, 900)
    core_s4 = r3_to_s3(core)
    a = 0.38 * eps
    core_angle = 2.0 * math.atan(a)
    lane_angle = 2.0 * math.atan(lane_radius)
    if lane_angle <= core_angle:
        raise ValueError("lane_radius must lie outside the core endpoint radius")

    pieces = [core_s4[:-1]]
    labels = [np.zeros(len(core_s4) - 1, dtype=np.int8)]
    pass_ids = [np.full(len(core_s4) - 1, -1, dtype=np.int16)]
    # First radial connector continues the core endpoint B in its +x direction.
    start_s = np.linspace(core_angle, lane_angle, 28, endpoint=False)
    pieces.append(meridian(start_s, 0.0))
    labels.append(np.ones(len(start_s), dtype=np.int8))
    pass_ids.append(np.zeros(len(start_s), dtype=np.int16))

    phi = np.linspace(0.0, math.pi, pass_count)
    arm_n = 220
    cap_n = max(12, int(36 / pass_count))
    for j in range(pass_count):
        s = np.linspace(lane_angle, math.pi - lane_angle, arm_n, endpoint=False)
        if j % 2:
            s = s[::-1]
        pieces.append(meridian(s, float(phi[j])))
        labels.append(np.ones(len(s), dtype=np.int8))
        pass_ids.append(np.full(len(s), j, dtype=np.int16))
        if j + 1 < pass_count:
            cap_s = math.pi - lane_angle if j % 2 == 0 else lane_angle
            cap = latitude_cap(cap_s, float(phi[j]), float(phi[j + 1]), cap_n)
            pieces.append(cap)
            labels.append(np.ones(len(cap), dtype=np.int8))
            pass_ids.append(np.full(len(cap), j, dtype=np.int16))

    # The final pass ends on the south latitude at phi=pi, which continues to A.
    end_s = np.linspace(lane_angle, core_angle, 28, endpoint=False)
    pieces.append(meridian(end_s, math.pi))
    labels.append(np.ones(len(end_s), dtype=np.int8))
    pass_ids.append(np.full(len(end_s), pass_count - 1, dtype=np.int16))

    raw_s4 = np.vstack(pieces)
    raw_labels = np.concatenate(labels)
    raw_pass_ids = np.concatenate(pass_ids)
    raw_L = float(geodesic_edges(raw_s4).sum())
    n = int(max(600, min(24000, math.ceil(n_per_len * raw_L))))
    s4, labels_out, pass_ids_out = resample_closed_s3_with_passes(raw_s4, raw_labels, raw_pass_ids, n)
    L = float(geodesic_edges(s4).sum())
    midpoint_separation = 2.0 * math.asin(math.sin(math.pi / 2.0) * math.sin(math.pi / (2.0 * (pass_count - 1))))
    return s4, labels_out, pass_ids_out, {
        "strand_model": "controlled_meridian_returns",
        "target_reached": 1.0,
        "far": float("nan"),
        "lateral_sep": float("nan"),
        "pass_count": float(pass_count),
        "lane_radius_r3": lane_radius,
        "lane_angle_s3": lane_angle,
        "midpoint_lane_separation": midpoint_separation,
        "n": float(n),
        "length": L,
    }


def split_energy(s4: np.ndarray, labels: np.ndarray, block: int = 768) -> Dict[str, float]:
    n = len(s4)
    edge = geodesic_edges(s4)
    S = np.concatenate([[0.0], np.cumsum(edge)])
    T = float(S[-1])
    w = 0.5 * (np.roll(edge, 1) + edge)
    acc = {"core_core": 0.0, "strand_strand": 0.0, "cross": 0.0}
    js_all = np.arange(n)
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        I = np.arange(i0, i1)[:, None]
        xi = s4[i0:i1]
        dots = np.clip(np.einsum("ik,jk->ij", xi, s4), -1.0, 1.0)
        theta = np.arccos(dots)
        fwd = S[js_all][None, :] - S[I]
        fwd = np.where(fwd >= 0.0, fwd, fwd + T)
        arc = np.minimum(fwd, T - fwd)
        sep = (js_all[None, :] - I) % n
        sep = np.minimum(sep, n - sep)
        mask = sep >= 2
        mask &= theta > 1e-14
        # Upper triangle only, then multiply by 2 below, matching C++.
        mask &= js_all[None, :] > I
        theta_safe = np.where(mask, theta, 1.0)
        arc_safe = np.where(mask, arc, 1.0)
        val = (1.0 / (theta_safe * theta_safe) - 1.0 / (arc_safe * arc_safe)) * (w[i0:i1, None] * w[None, :])
        pair = labels[i0:i1, None] + labels[None, :]
        for key, pval in (("core_core", 0), ("cross", 1), ("strand_strand", 2)):
            acc[key] += float(np.sum(val[mask & (pair == pval)]))
    for k in acc:
        acc[k] *= 2.0
    acc["total"] = acc["core_core"] + acc["strand_strand"] + acc["cross"]
    acc["length"] = T
    acc["core_fraction_vertices"] = float(np.mean(labels == 0))
    return acc


def pass_energy_matrix(s4: np.ndarray, labels: np.ndarray, pass_ids: np.ndarray,
                       pass_count: int, block: int = 768) -> np.ndarray:
    """Return strand energy by pass pair, using the same quadrature as split_energy.

    Diagonal entries are a pass's self-energy.  Off-diagonal entries contain
    the complete ordered-pair contribution of the two distinct passes once.
    """
    n = len(s4)
    edge = geodesic_edges(s4)
    S = np.concatenate([[0.0], np.cumsum(edge)])
    T = float(S[-1])
    w = 0.5 * (np.roll(edge, 1) + edge)
    out = np.zeros((pass_count, pass_count), dtype=float)
    js_all = np.arange(n)
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        I = np.arange(i0, i1)[:, None]
        dots = np.clip(np.einsum("ik,jk->ij", s4[i0:i1], s4), -1.0, 1.0)
        theta = np.arccos(dots)
        fwd = S[js_all][None, :] - S[I]
        fwd = np.where(fwd >= 0.0, fwd, fwd + T)
        arc = np.minimum(fwd, T - fwd)
        sep = (js_all[None, :] - I) % n
        sep = np.minimum(sep, n - sep)
        mask = (js_all[None, :] > I) & (sep >= 2) & (theta > 1e-14)
        pi = pass_ids[i0:i1, None]
        pj = pass_ids[None, :]
        mask &= (labels[i0:i1, None] == 1) & (labels[None, :] == 1)
        mask &= (pi >= 0) & (pj >= 0)
        theta_safe = np.where(mask, theta, 1.0)
        arc_safe = np.where(mask, arc, 1.0)
        val = 2.0 * (1.0 / (theta_safe * theta_safe) - 1.0 / (arc_safe * arc_safe))
        val *= w[i0:i1, None] * w[None, :]
        ii, jj = np.nonzero(mask)
        lo = np.minimum(pi[ii, 0], pj[0, jj])
        hi = np.maximum(pi[ii, 0], pj[0, jj])
        np.add.at(out, (lo, hi), val[ii, jj])
    return out


def pass_statistics(matrix: np.ndarray, length: float) -> Dict[str, float]:
    diagonal = np.diag(matrix)
    shares = diagonal + 0.5 * (matrix.sum(axis=0) + matrix.sum(axis=1) - 2.0 * diagonal)
    adjacent = np.diag(matrix, k=1)
    return {
        "pass_self_raw_mean": float(np.mean(diagonal)),
        "pass_self_raw_median": float(np.median(diagonal)),
        "pass_share_raw_mean": float(np.mean(shares)),
        "pass_share_raw_median": float(np.median(shares)),
        "pass_share_q_mean": float(np.mean(shares) / (length * length)),
        "adjacent_pass_raw_mean": float(np.mean(adjacent)),
        "adjacent_pass_raw_median": float(np.median(adjacent)),
    }


def diagnostics(s4: np.ndarray, labels: np.ndarray, eps: float) -> Dict[str, float]:
    r3 = s3_to_r3(s4)
    core_r = float(np.max(np.linalg.norm(r3[labels == 0], axis=1)))
    edge = geodesic_edges(s4)
    u = normalize(np.roll(s4, -1, axis=0) - s4)
    v = normalize(s4 - np.roll(s4, 1, axis=0))
    angles = np.arccos(np.clip(np.einsum("ij,ij->i", v, u), -1.0, 1.0))
    jumps = np.flatnonzero(labels != np.roll(labels, 1))
    jmax = float(np.max(angles[jumps])) if len(jumps) else 0.0
    return {
        "core_radius_r3": core_r,
        "core_radius_over_eps": core_r / eps,
        "max_edge": float(edge.max()),
        "mean_edge": float(edge.mean()),
        "max_turn_angle": float(angles.max()),
        "max_junction_angle": jmax,
    }


def fit_models(rows: List[Dict[str, float]], family: str, term: str) -> Dict[str, float]:
    xs = np.array([r["length"] for r in rows if r["family"] == family], float)
    ys = np.array([r[term + "_q"] for r in rows if r["family"] == family], float)
    out: Dict[str, float] = {}
    A = np.column_stack([np.ones_like(xs), np.log(xs)])
    c = np.linalg.lstsq(A, ys, rcond=None)[0]
    pred = A @ c
    out["const_log_intercept"] = float(c[0])
    out["const_log_slope"] = float(c[1])
    out["const_log_rmse"] = float(np.sqrt(np.mean((ys - pred) ** 2)))
    out["constant_mean"] = float(np.mean(ys))
    out["constant_rmse"] = float(np.sqrt(np.mean((ys - np.mean(ys)) ** 2)))
    positive = ys > 0
    if np.count_nonzero(positive) >= 3:
        B = np.column_stack([np.ones(np.count_nonzero(positive)), np.log(xs[positive])])
        d = np.linalg.lstsq(B, np.log(ys[positive]), rcond=None)[0]
        out["power_exponent"] = float(d[1])
        out["power_prefactor"] = float(math.exp(d[0]))
        out["power_log_rmse"] = float(np.sqrt(np.mean((np.log(ys[positive]) - B @ d) ** 2)))
    else:
        out["power_exponent"] = float("nan")
        out["power_prefactor"] = float("nan")
        out["power_log_rmse"] = float("nan")
    return out


def fit_against_pass_count(rows: List[Dict[str, float]], family: str, field: str) -> Dict[str, float]:
    rr = [r for r in rows if r["family"] == family]
    xs = np.array([r["pass_count"] for r in rr], float)
    ys = np.array([r[field] for r in rr], float)
    A = np.column_stack([np.ones_like(xs), np.log(xs)])
    c = np.linalg.lstsq(A, ys, rcond=None)[0]
    positive = ys > 0
    B = np.column_stack([np.ones(np.count_nonzero(positive)), np.log(xs[positive])])
    d = np.linalg.lstsq(B, np.log(ys[positive]), rcond=None)[0]
    return {
        "log_intercept": float(c[0]),
        "log_slope": float(c[1]),
        "log_rmse": float(np.sqrt(np.mean((ys - A @ c) ** 2))),
        "power_prefactor": float(math.exp(d[0])),
        "power_exponent": float(d[1]),
        "power_log_rmse": float(np.sqrt(np.mean((np.log(ys[positive]) - B @ d) ** 2))),
    }


def plot(rows: List[Dict[str, float]], out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terms = [("core_core_q", "core-core"), ("strand_strand_q", "strand-strand"), ("cross_q", "cross")]
    styles = {"trefoil": "-", "unknot": "--"}
    colors = {"core_core_q": "tab:blue", "strand_strand_q": "tab:green", "cross_q": "tab:red"}
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for family in ("trefoil", "unknot"):
        rr = sorted([r for r in rows if r["family"] == family], key=lambda x: x["length"])
        x = np.log([r["length"] for r in rr])
        for term, label in terms:
            ax.plot(x, [r[term] for r in rr], styles[family], color=colors[term],
                    marker="o" if family == "trefoil" else "s", ms=4,
                    label=f"{label} ({family})")
    ax.set_xlabel("log S3 length")
    ax.set_ylabel("contribution to E / L^2")
    ax.set_title("Long-strand decomposition of S3 O'Hara quantity energy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pass_statistics(rows: List[Dict[str, float]], out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [
        ("pass_self_raw_mean", "pass self-energy"),
        ("adjacent_pass_raw_mean", "adjacent-pair energy"),
        ("pass_share_raw_mean", "pass allocated share"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for family, style in (("trefoil", "o-"), ("unknot", "s--")):
        rr = sorted((r for r in rows if r["family"] == family), key=lambda r: r["pass_count"])
        x = [r["pass_count"] for r in rr]
        for ax, (field, title) in zip(axes, fields):
            ax.plot(x, [r[field] for r in rr], style, label=family)
            ax.set_title(title)
            ax.set_xlabel("number of passes k")
            ax.grid(alpha=0.25)
    axes[0].set_ylabel("raw energy contribution")
    axes[0].legend()
    fig.suptitle("Controlled near-return strand: per-pass diagnostics")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_multi_return_summary(rows: List[Dict[str, float]], pass_fits: Dict[str, Dict[str, Dict[str, float]]],
                               path: str) -> None:
    with open(path, "w") as f:
        f.write("# Controlled Near-Return Experiment\n\n")
        f.write("The strand consists of an even number k of alternating, near-great-circle meridian passes. "
                "Adjacent passes are joined by short latitude caps at fixed S3 distance from the poles. "
                "At the midpoint, adjacent-lane separation is 2 asin(sin(pi/(2(k-1)))), asymptotic to pi/k.\n\n")
        for family in ("trefoil", "unknot"):
            rr = sorted((r for r in rows if r["family"] == family), key=lambda r: r["pass_count"])
            first, last = rr[0], rr[-1]
            f.write(f"## {family}\n\n")
            f.write(f"L: {first['length']:.3f} at k={int(first['pass_count'])} to "
                    f"{last['length']:.3f} at k={int(last['pass_count'])}.  "
                    f"E/L^2: {first['total_q']:.6g} to {last['total_q']:.6g}.\n\n")
            for field, label in (("pass_self_raw_mean", "mean pass self-energy"),
                                 ("adjacent_pass_raw_mean", "mean adjacent-pass energy"),
                                 ("pass_share_raw_mean", "mean allocated pass share")):
                fit = pass_fits[family][field]
                f.write(f"{label}: power fit k^{fit['power_exponent']:.3f}; "
                        f"linear-in-log(k) slope {fit['log_slope']:.3f}.\n\n")
        f.write("## Numerical caveat\n\n")
        f.write("The trefoil core still has a maximum stitch angle near 1.45 radians. "
                "This makes core and cross contributions stitch-sensitive. The strand-strand and pass-pair "
                "measurements are the relevant diagnostics for the controlled-return mechanism.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", nargs="+", type=float, default=None,
                    help="Optional target S3 lengths. Values outside the one-excursion range are clipped and flagged.")
    ap.add_argument("--far-values", nargs="+", type=float, default=[0.6, 1, 2, 5, 10, 50, 500],
                    help="Default sweep: finite stereographic distance of the pole excursion.")
    ap.add_argument("--eps", type=float, default=0.18)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--min-far", type=float, default=0.6)
    ap.add_argument("--max-far", type=float, default=500.0)
    ap.add_argument("--lateral-sep", type=float, default=0.55)
    ap.add_argument("--n-per-length", type=float, default=80.0)
    ap.add_argument("--pass-counts", nargs="+", type=int, default=None,
                    help="Even numbers of controlled near-great-circle passes. Enables the multi-return experiment.")
    ap.add_argument("--lane-radius", type=float, default=0.30,
                    help="R3 radius of the south/north latitude connectors; must exceed the core endpoint radius.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows: List[Dict[str, float]] = []
    if args.pass_counts is not None:
        sweep = [("passes", x) for x in args.pass_counts]
    elif args.lengths is not None:
        sweep = [("length", x) for x in args.lengths]
    else:
        sweep = [("far", x) for x in args.far_values]
    matrices: Dict[str, np.ndarray] = {}
    for sweep_kind, sweep_value in sweep:
        for family in ("trefoil", "unknot"):
            if sweep_kind == "passes":
                s4, labels, pass_ids, info = multi_return_curve(
                    family, int(sweep_value), args.eps, args.lane_radius, args.n_per_length
                )
                target = float("nan")
                tag = f"{family}_k{int(sweep_value)}"
            elif sweep_kind == "length":
                s4, labels, info = curve_for_length(
                    family, sweep_value, args.eps, args.lateral_sep, args.min_far,
                    args.max_far, args.n_per_length
                )
                target = sweep_value
                tag = f"{family}_L{target:g}"
            else:
                s4, labels, info = curve_for_far(
                    family, sweep_value, args.eps, args.lateral_sep, args.n_per_length
                )
                target = float("nan")
                tag = f"{family}_far{sweep_value:g}"
            e = split_energy(s4, labels)
            d = diagnostics(s4, labels, args.eps)
            row: Dict[str, float] = {"family": family, "target_length": target, **info, **e, **d}
            if sweep_kind == "passes":
                matrix = pass_energy_matrix(s4, labels, pass_ids, int(sweep_value))
                row.update(pass_statistics(matrix, row["length"]))
                matrices[tag] = matrix
            for key in ("core_core", "strand_strand", "cross", "total"):
                row[key + "_q"] = row[key] / (row["length"] * row["length"])
            rows.append(row)
            write_s4(os.path.join(args.out, tag + ".s4"), s4, labels)
            write_vect(os.path.join(args.out, tag + ".vect"), s3_to_r3(s4))
            reach = "" if row["target_reached"] else " clipped"
            if sweep_kind == "passes":
                print("%-15s L=%7.3f n=%4d sep=%6.3g  Eq=% .6g  core=% .3g strand=% .3g cross=% .3g  pass-self=% .3g adjacent=% .3g" %
                      (tag, row["length"], int(row["n"]), row["midpoint_lane_separation"], row["total_q"],
                       row["core_core_q"], row["strand_strand_q"], row["cross_q"],
                       row["pass_self_raw_mean"], row["adjacent_pass_raw_mean"]))
            else:
                print("%-15s L=%7.3f n=%4d far=%8.3g%s  Eq=% .6g  core=% .3g strand=% .3g cross=% .3g  jangle=%.3f" %
                      (tag, row["length"], int(row["n"]), row["far"], reach, row["total_q"],
                       row["core_core_q"], row["strand_strand_q"], row["cross_q"], row["max_junction_angle"]))

    csv_path = os.path.join(args.out, "decomposition.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fits: Dict[str, Dict[str, Dict[str, float]]] = {}
    for fam in ("trefoil", "unknot"):
        fits[fam] = {}
        for term in ("core_core", "strand_strand", "cross", "total"):
            fits[fam][term] = fit_models(rows, fam, term)
    json_path = os.path.join(args.out, "summary.json")
    with open(json_path, "w") as f:
        payload = {"args": vars(args), "fits": fits, "rows": rows}
        if args.pass_counts is not None:
            payload["pass_fits"] = {
                fam: {field: fit_against_pass_count(rows, fam, field)
                      for field in ("pass_self_raw_mean", "adjacent_pass_raw_mean", "pass_share_raw_mean")}
                for fam in ("trefoil", "unknot")
            }
        json.dump(payload, f, indent=2)

    plot_path = os.path.join(args.out, "decomposition_vs_logL.png")
    plot(rows, plot_path)
    if args.pass_counts is not None:
        matrix_path = os.path.join(args.out, "pass_energy_matrices.npz")
        np.savez(matrix_path, **matrices)
        pass_plot_path = os.path.join(args.out, "per_pass_vs_k.png")
        plot_pass_statistics(rows, pass_plot_path)
        summary_path = os.path.join(args.out, "summary.md")
        pass_fits = {
            fam: {field: fit_against_pass_count(rows, fam, field)
                  for field in ("pass_self_raw_mean", "adjacent_pass_raw_mean", "pass_share_raw_mean")}
            for fam in ("trefoil", "unknot")
        }
        write_multi_return_summary(rows, pass_fits, summary_path)
        print(f"Wrote {matrix_path}")
        print(f"Wrote {pass_plot_path}")
        print(f"Wrote {summary_path}")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print("\nLog-fit slopes for E/L^2 contributions:")
    for fam in ("trefoil", "unknot"):
        bits = [f"{term}={fits[fam][term]['const_log_slope']:.6g}" for term in ("core_core", "strand_strand", "cross", "total")]
        print(f"  {fam}: " + ", ".join(bits))


if __name__ == "__main__":
    main()
