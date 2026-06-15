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


def _interp(a, b, k):
    """k points strictly between a and b (endpoints excluded)."""
    return [(a + (b - a) * (j / (k + 1.0))).tolist() for j in range(1, k + 1)]


def connect_sum(specs, n_total):
    """specs = [(p1,q1), (p2,q2), …]  →  (N,3) closed connect-sum polygon."""
    M = len(specs)
    sep = 4.0
    n_each = max(40, int(n_total * 0.82) // M)
    knots = []
    for i, (p, q) in enumerate(specs):
        K = torus_knot_r3(p, q, n_each) * 1.4
        K[:, 0] += i * sep
        knots.append(K)

    zmin = min(K[:, 2].min() for K in knots)
    zfloor, zret = zmin - 1.2, zmin - 2.6        # forward / return bridge planes

    # Open each knot at its lowest vertex: arc[0] = entry, arc[-1] = exit.
    arcs = []
    for K in knots:
        N = len(K)
        bi = int(np.argmin(K[:, 2]))
        arcs.append(K[[(bi + 1 + j) % N for j in range(N)]])

    nb = max(6, int(n_total * 0.18) // M)
    loop = []
    for i in range(M):
        arc = arcs[i]
        loop.extend(arc.tolist())                # traverse the whole knot
        u = arc[-1]                              # exit of knot i
        w = arcs[(i + 1) % M][0]                  # entry of next knot
        z = zret if i == M - 1 else zfloor        # return bridge dips deeper
        a, b = u.copy(), w.copy()
        a[2] = z
        b[2] = z
        loop += _interp(u, a, 2) + [a.tolist()]
        loop += _interp(a, b, nb if i < M - 1 else 2 * nb) + [b.tolist()]
        loop += _interp(b, w, 2)                  # next arc starts exactly at w
    return np.array(loop)


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
