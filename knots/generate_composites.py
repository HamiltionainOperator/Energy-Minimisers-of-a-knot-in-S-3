import numpy as np
import os
import argparse


# ───────────────────────────────────────────────────────────────────────────
# General connect-sum of torus knots:  T(p1,q1) # T(p2,q2) # …
#
# Each T(pi,qi) is laid out in its own column along x, cut open at its bottom,
# and the columns are chained into one closed loop by bridges routed BELOW all
# the knots — forward bridges in one low plane, the single return bridge in a
# deeper plane — so no bridge can pass through a knot or cross another bridge.
# The result is a clean connect sum; its determinant is the product of the
# components' determinants (verify with analysis/knot_check.py).
# ───────────────────────────────────────────────────────────────────────────

def torus_knot_r3(p, q, n):
    """A (p,q) torus knot in R³, centred at the origin and scaled to unit radius."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    R, r = 2.0, 0.85
    c = R + r * np.cos(q * t)
    P = np.column_stack([c * np.cos(p * t), c * np.sin(p * t), r * np.sin(q * t)])
    P -= P.mean(0)
    P /= np.max(np.linalg.norm(P, axis=1))
    return P


def _hermite(p0, m0, p1, m1, k):
    """k interior points of the cubic Hermite spline p0→p1 with end tangents m0,m1.

    Tangent-matched so the curve leaves p0 and enters p1 smoothly (G¹) — no sharp
    corner where a bridge meets a knot, which is what blew the gradient up before.
    """
    pts = []
    for j in range(1, k + 1):
        t = j / (k + 1.0)
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        pts.append((h00*p0 + h10*m0 + h01*p1 + h11*m1).tolist())
    return pts


def _seg_polyline_dist(p, q):
    """Crude min distance between two point-lists (used to detect a crossing)."""
    P, Q = np.asarray(p), np.asarray(q)
    return float(np.min(np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)))


def resample_uniform(loop, N):
    """Resample a closed polygon to N points equally spaced in arc length.

    Critical: a uniform edge length means knot segments and bridge segments are
    the same size, so the discrete energy has no near-zero edge / huge-curvature
    spikes — that mismatch is what produced the ~10⁶ initial gradient.
    """
    loop = np.asarray(loop, dtype=float)
    closed = np.vstack([loop, loop[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    targets = np.linspace(0.0, total, N, endpoint=False)
    idx = np.searchsorted(cum, targets, side="right") - 1
    idx = np.clip(idx, 0, len(loop) - 1)
    out = np.empty((N, 3))
    for k, (s, j) in enumerate(zip(targets, idx)):
        f = (s - cum[j]) / (seg[j] if seg[j] > 1e-12 else 1.0)
        out[k] = loop[j] + (loop[(j + 1) % len(loop)] - loop[j]) * f
    return out


def _open_arc(K, toward, gw):
    """Open knot K by REMOVING a short arc (2·gw+1 vertices) at the vertex facing
    `toward`. Returns the remaining open polyline; arc[0] and arc[-1] are the two
    cut ends, offset by the removed arc — the natural width of the connect band.
    """
    N = len(K)
    p = int(np.argmin(np.linalg.norm(K - toward, axis=1)))
    keep = [(p + gw + 1 + j) % N for j in range(N - (2 * gw + 1))]
    return K[keep]


def connect_sum(specs, n_total):
    """specs = [(p1,q1), (p2,q2), …]  →  (N,3) closed connect-sum polygon.

    Textbook construction: each summand is OPENED by removing a short arc on the
    side facing its neighbour; the consecutive cut ends are joined by short,
    tangent-matched bridges that form a flat, untwisted band (no bow). For each
    join we pick the neighbour's traversal orientation that keeps the two band
    strands from crossing. Finally the loop is resampled to uniform arc length.
    """
    M = len(specs)
    sep = 2.4
    n_each = max(80, int(n_total * 0.85) // M)
    gw = max(1, n_each // 50)                       # half-width of removed arc
    knots = []
    for i, (p, q) in enumerate(specs):
        K = torus_knot_r3(p, q, n_each)
        K[:, 0] += i * sep
        knots.append(K)
    centers = [K.mean(0) for K in knots]

    arcs = [_open_arc(knots[i], centers[(i + 1) % M], gw) for i in range(M)]

    nb = max(8, int(n_total * 0.12) // M)
    loop = []
    for i in range(M):
        arc = arcs[i]
        nxt = arcs[(i + 1) % M]
        # Bridge from this arc's end (u) to the next arc's start (w). Try the
        # next arc both ways and keep the orientation whose bridge stays farther
        # from the *other* end of the band (no crossing → flat, untwisted band).
        u, u_prev = arc[-1], arc[-2]
        cand = []
        for rev in (False, True):
            a2 = nxt[::-1] if rev else nxt
            w, w_next = a2[0], a2[1]
            t_u = (u - u_prev); t_u /= (np.linalg.norm(t_u) + 1e-9)
            t_w = (w_next - w); t_w /= (np.linalg.norm(t_w) + 1e-9)
            L = np.linalg.norm(w - u) + 1e-9
            br = _hermite(u, t_u * L, w, t_w * L, nb)
            # separation from the band's other strand (this arc's start side)
            sep_score = _seg_polyline_dist(br, [arc[0].tolist(), nxt[-1].tolist()])
            cand.append((sep_score, rev, br, a2))
        cand.sort(reverse=True)                     # most-separated orientation wins
        _, _, br, a2 = cand[0]
        arcs[(i + 1) % M] = a2                       # commit chosen orientation
        loop.extend(arc.tolist())
        loop.extend(br)
    return resample_uniform(np.array(loop), n_total)


def parse_spec(s):
    """'2,3' or '2x3' or '2_3'  →  (2, 3)."""
    for sep in (",", "x", "_"):
        if sep in s:
            a, b = s.split(sep)
            return int(a), int(b)
    raise argparse.ArgumentTypeError(f"bad torus spec {s!r}; use e.g. 2,3")


def generate_granny_knot(N=5000):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = -0.22 * np.cos(t) - 1.28 * np.sin(t) - 0.44 * np.cos(3*t) - 0.78 * np.sin(3*t)
    y = -0.10 * np.cos(2*t) - 0.27 * np.sin(2*t) + 0.38 * np.cos(4*t) + 0.46 * np.sin(4*t)
    z =  0.70 * np.cos(3*t) - 0.40 * np.sin(3*t)
    return np.column_stack([x, y, z]) * 20.0

def generate_square_knot(N=5000):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = -0.22 * np.cos(t) - 1.28 * np.sin(t) - 0.44 * np.cos(3*t) - 0.78 * np.sin(3*t)
    y =  0.11 * np.cos(t) - 0.43 * np.sin(3*t) + 0.34 * np.cos(5*t) - 0.39 * np.sin(5*t)
    z =  0.70 * np.cos(3*t) - 0.40 * np.sin(3*t) + 0.18 * np.cos(5*t) - 0.09 * np.sin(5*t)
    return np.column_stack([x, y, z]) * 20.0

def generate_granny_left(N=5000):
    # Mirror of the Granny Knot (invert Z axis)
    pts = generate_granny_knot(N)
    pts[:, 2] = -pts[:, 2]
    return pts

def write_vect(filename, pts):
    with open(filename, 'w') as f:
        f.write("1\n")
        f.write(f"{len(pts)}\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate composite knots: named presets, or an arbitrary "
                    "connect sum of torus knots via --connect.")
    parser.add_argument('--n', type=int, default=1000,
                        help='approximate total number of points')
    parser.add_argument('--type', type=str,
                        choices=['granny', 'square', 'granny_left'],
                        help='named composite preset')
    parser.add_argument('--connect', nargs='+', type=parse_spec, metavar='p,q',
                        help='connect-sum of torus knots, e.g. --connect 2,3 2,5')
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    if not args.type and not args.connect:
        parser.error("give --type or --connect")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    if args.connect:
        pts = connect_sum(args.connect, args.n)
        label = " # ".join(f"T({p},{q})" for p, q in args.connect)
    elif args.type == 'granny':
        pts = generate_granny_knot(args.n)
        label = 'granny'
    elif args.type == 'square':
        pts = generate_square_knot(args.n)
        label = 'square'
    elif args.type == 'granny_left':
        pts = generate_granny_left(args.n)
        label = 'granny_left'

    write_vect(args.out, pts)
    print(f"Generated {label}  (N={len(pts)})  →  {args.out}")
