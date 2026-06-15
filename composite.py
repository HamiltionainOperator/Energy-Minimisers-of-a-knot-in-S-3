#!/usr/bin/env python3
"""
composite.py — connect two (or more) energy-minimised .vect files with a
short geodesic bridge to form a connected sum (granny, square, etc.).

Examples:
  Granny knot  3_1 # 3_1
    python3 composite.py output/T2_3/T2_3_s3.vect output/T2_3/T2_3_s3.vect \
        --n 2000 --bridge-len 0.08 --bridge-points 30 --mirror none \
        --out output/granny_T23_T23/granny_T23_T23

  Square knot  3_1 # mirror(3_1)
    python3 composite.py output/T2_3/T2_3_s3.vect output/T2_3/T2_3_s3.vect \
        --n 2000 --bridge-len 0.08 --bridge-points 30 --mirror second \
        --out output/square_T23_T23/square_T23_T23
"""
import argparse
import numpy as np


# ============================================================================
# I/O
# ============================================================================


def load_vect(path):

    """

    Read a .vect file of the form

        1

        N

        x y z

        x y z

        ...

    and lift the curve to S^3 by adding

        w = sqrt(1 - x^2 - y^2 - z^2)

    If the file already contains 4 coordinates, keep them.

    """

    rows = []

    with open(path, "r") as f:

        lines = [line.strip() for line in f if line.strip()]

    for line in lines[2:]:  # skip header lines

        parts = line.split()

        if len(parts) == 3:
                x, y, z = map(float, parts)
                rows.append([x,x,z,0.0])

    # Input file contains ordinary R^3 coordinates.

    # Embed into R^4 and project to S^3 later.





        elif len(parts) == 4:

            rows.append(list(map(float, parts)))

    if not rows:

        raise ValueError(f"No coordinates found in {path}")

    return np.asarray(rows, dtype=float)

def write_obj(P, path):
    """Write a closed polyline as an OBJ with one 'l' (line) element."""
    with open(path, 'w') as f:
        f.write("# composite polyline, {} vertices\n".format(len(P)))
        for v in P:
            f.write("v {:.10f} {:.10f} {:.10f} {:.10f}\n".format(*v))
        # closed line: indices 1..n then back to 1
        f.write(" ".join(["l"] + ["{}".format(i + 1) for i in range(len(P))]
                         + ["1\n"]))


def write_vect(P, path):

    """

    Write VECT format expected by energy_s3:

    1

    N

    x y z w

    ...

    """

    with open(path, "w") as f:

        f.write("1\n")

        f.write(f"{len(P)}\n")

        for row in P:

            f.write(

                f"{row[0]:.10f} "

                f"{row[1]:.10f} "

                f"{row[2]:.10f} "

                f"{row[3]:.10f}\n"

            )

def project_to_s3(P):
    """Project each row to the unit 3-sphere in R^4."""
    n = np.linalg.norm(P, axis=1, keepdims=True)
    return P / n


def normalise_rms(P, target=1.0):
    """Rescale so the RMS radius (in R^4) equals `target`."""
    rms = np.sqrt(np.mean(np.sum(P ** 2, axis=1)))
    if rms == 0:
        return P
    return P * (target / rms)


def arclength_resample(P, n_out):
    """Resample a polyline in R^4 to n_out vertices by chordal arclength."""
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    s_total = s[-1]
    if s_total == 0:
        return np.tile(P[0], (n_out, 1))
    s_out = np.linspace(0, s_total, n_out, endpoint=False)
    idx = np.searchsorted(s, s_out, side='right') - 1
    idx = np.clip(idx, 0, len(P) - 2)
    denom = s[idx + 1] - s[idx]
    denom = np.where(denom == 0, 1.0, denom)
    t = (s_out - s[idx]) / denom
    return (1 - t)[:, None] * P[idx] + t[:, None] * P[idx + 1]


def geodesic_arc(p0, p1, n):
    """Short great-circle arc on S^3 from unit p0 to unit p1, n points."""
    p0 = p0 / np.linalg.norm(p0)
    p1 = p1 / np.linalg.norm(p1)
    cos_t = np.clip(np.dot(p0, p1), -1.0, 1.0)
    t = np.arccos(cos_t)
    if t < 1e-9:
        return np.tile(p0, (n, 1))
    ts = np.linspace(0.0, 1.0, n)
    s0 = np.sin((1 - ts) * t) / np.sin(t)
    s1 = np.sin(ts * t) / np.sin(t)
    return s0[:, None] * p0 + s1[:, None] * p1


# ============================================================================
# 4D rotations and reflections
# ============================================================================

def rotation_taking_a_to_b(a, b):
    """
    Return a 4x4 rotation R such that R @ a == b (a, b as column unit
    vectors). The rotation lies in the 2-plane spanned by a and b.
    Apply to row-vector polyline as: P @ R.T
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cos_t = np.clip(np.dot(a, b), -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-12:
        return np.eye(4)
    sin_t = np.sin(theta)
    u = a
    v = (b - cos_t * a) / sin_t
    c, s = np.cos(theta), np.sin(theta)
    R = (np.eye(4)
         + (c - 1.0) * (np.outer(u, u) + np.outer(v, v))
         + s * (np.outer(v, u) - np.outer(u, v)))
    return R


def mirror_factor(P, axis):
    """
    Apply an orientation-reversing isometry of R^4 that flips the sign of
    coordinate `axis` (0..3).  R = I - 2 e_axis e_axis^T  (Householder).
    Apply to row vectors as: P @ R.T.
    """
    axis = axis % 4
    R = np.eye(4)
    R[axis, axis] = -1.0
    return P @ R.T


# ============================================================================
# Alignment to a canonical basepoint
# ============================================================================

def align_to_basepoint(P, basepoint_axis=2, tangent_axis=0):
    """
    Roll and rotate P so that:
      - the point with minimum coordinate along basepoint_axis becomes
        index 0 and is mapped to (0, ..., -1, 0, ...) (south pole of S^3
        along `basepoint_axis`),
      - the tangent at index 0 is parallel to +e_{tangent_axis} and
        lies in the tangent plane of S^3 at the south pole.
    """
    basepoint_axis = basepoint_axis % 4
    tangent_axis = tangent_axis % 4

    # 1. Roll so the southernmost point is at index 0.
    k = np.argmin(P[:, basepoint_axis])
    P = np.roll(P, -k, axis=0)

    # 2. Rotate the whole polyline so P[0] lands at the south pole.
    target_p = np.zeros(4)
    target_p[basepoint_axis] = -1.0
    R_pos = rotation_taking_a_to_b(P[0], target_p)
    P = P @ R_pos.T

    # 3. Align the tangent in the tangent plane at the south pole.
    #    The radial direction at the south pole is the point itself.
    radial = np.zeros(4)
    radial[basepoint_axis] = -1.0
    t_curr = P[1] - P[0]
    t_curr = t_curr - np.dot(t_curr, radial) * radial
    nt = np.linalg.norm(t_curr)
    if nt < 1e-12:
        return P  # degenerate
    t_curr = t_curr / nt

    target_t = np.zeros(4)
    target_t[tangent_axis] = 1.0
    target_t = target_t - np.dot(target_t, radial) * radial
    target_t = target_t / np.linalg.norm(target_t)

    R_tan = rotation_taking_a_to_b(t_curr, target_t)
    P = P @ R_tan.T
    return P


# ============================================================================
# Diagnostics
# ============================================================================

def nearest_neighbor_distance(P, k=5):
    """
    Smallest non-self chordal distance in the polyline.  Skip each
    vertex's k nearest neighbours on either side along the closed curve
    to avoid the trivial small-edge answer.
    """
    n = len(P)
    d_min = np.inf
    for i in range(n):
        js = []
        for off in range(1, k + 1):
            js.append((i - off) % n)
            js.append((i + off) % n)
        for j in js:
            if j == i:
                continue
            d = np.linalg.norm(P[i] - P[j])
            if 0 < d < d_min:
                d_min = d
    return d_min


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+",
                    help="Two or more .vect files (the factors).")
    ap.add_argument("--n", type=int, default=2000,
                    help="Total vertex count of the output.")
    ap.add_argument("--bridge-len", type=float, default=0.08,
                    help="Approx. chordal length of each bridge arc.")
    ap.add_argument("--bridge-points", type=int, default=30,
                    help="Number of vertices per bridge arc.")
    ap.add_argument("--axis", type=int, default=2,
                    help="Coordinate axis used as the 'pole' (0..3).")
    ap.add_argument("--tangent-axis", type=int, default=0,
                    help="Coordinate axis the aligned tangent should be "
                         "parallel to (0..3).")
    ap.add_argument("--mirror", choices=["none", "second", "all", "alternating"],
                    default="none",
                    help="Which factors to mirror: 'none' = granny, "
                         "'second' = square knot, 'alternating' = "
                         "mirror every other factor.")
    ap.add_argument("--mirror-axis", type=int, default=2,
                    help="Which coordinate axis to flip for mirroring "
                         "(default 2: (x,y,z,w) -> (x,y,-z,w)).")
    ap.add_argument("--out", required=True,
                    help="Output path prefix (no extension).")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        ap.error("Need at least 2 factors.")

    # Load and project factors to S^3
    factors = [project_to_s3(load_vect(p)) for p in args.inputs]

    # Mirror mask
    n_factors = len(factors)
    if args.mirror == "none":
        mask = [False] * n_factors
    elif args.mirror == "all":
        mask = [True] * n_factors
    elif args.mirror == "second":
        mask = [False] * n_factors
        if n_factors >= 2:
            mask[1] = True
    elif args.mirror == "alternating":
        mask = [(i % 2 == 1) for i in range(n_factors)]
    else:
        raise ValueError(args.mirror)

    # Apply mirror, then align
    aligned = []
    for i, P in enumerate(factors):
        if mask[i]:
            P = mirror_factor(P, axis=args.mirror_axis)
            P = project_to_s3(P)
        P = align_to_basepoint(P, basepoint_axis=args.axis,
                               tangent_axis=args.tangent_axis)
        aligned.append(P)

    # Place each factor on alternating sides of the equator
    sep = args.bridge_len / 2.0
    placed = []
    for i, P in enumerate(aligned):
        sign = +1.0 if i % 2 == 0 else -1.0
        P = P.copy()
        P[:, args.axis] += sign * sep
        P = project_to_s3(P)
        placed.append(P)

    # Concatenate with geodesic bridges
    parts = [placed[0]]
    for i in range(1, len(placed)):
        bridge = geodesic_arc(parts[-1][-1], placed[i][0],
                              args.bridge_points)
        parts.append(bridge)
        parts.append(placed[i])

    combined = np.concatenate(parts, axis=0)

    # Resample, project, renormalise
    combined = project_to_s3(combined)
    combined = arclength_resample(combined, args.n)
    combined = project_to_s3(combined)
    combined = normalise_rms(combined, target=1.0)

    write_obj(combined, args.out + ".obj")
    write_vect(combined, args.out + ".vect")

    d_min = nearest_neighbor_distance(combined)
    print("Written {} vertices -> {}.{{obj,vect}}"
          .format(len(combined), args.out))
    print("Mirror mask:       {}".format(mask))
    print("Min strand gap:    {:.6f}".format(d_min))
    print("Bridge chord:      {:.6f} (target {})"
          .format(args.bridge_len, args.bridge_len))


if __name__ == "__main__":
    main()