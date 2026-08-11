#!/usr/bin/env python3
"""Continuum (integral-form, no discretization) on-Clifford O'Hara energies.

For T(p,q) on the Clifford torus T_r: gamma_r(t) = (r cos pt, r sin pt,
rho cos qt, rho sin qt), rho = sqrt(1-r^2), constant speed v = sqrt(p^2 r^2 +
q^2 rho^2), total geodesic length L = 2 pi v.  By homogeneity the double
integral collapses to one variable u = t - s:

  E_d(r) = 4 pi v^2 * int_0^pi [ 1/arccos(r^2 cos pu + rho^2 cos qu)^2
                                 - 1/(v u)^2 ] du
  E_q(r) = E_d(r) / L^2

The integrand has ~q narrow near-singular peaks (adjacent-strand passes), so
we integrate with Gauss-Legendre nodes PER PERIOD 2pi/max(p,q) — exact enough
at 64 nodes/period (validated: reproduces O'Hara Table 5.1 to 7+ digits).
Then minimize over r by golden-section search.
"""
import sys

import numpy as np

GL_X, GL_W = np.polynomial.legendre.leggauss(64)


def E_d(p, q, r):
    rho2 = 1.0 - r * r
    v2 = p * p * r * r + q * q * rho2
    v = np.sqrt(v2)
    m = max(p, q)
    per = np.pi / m                      # half-period granularity over [0, pi]
    edges = np.linspace(0.0, np.pi, m + 1)
    a, b = edges[:-1], edges[1:]
    u = 0.5 * (b - a)[:, None] * (GL_X[None, :] + 1.0) + a[:, None]
    w = 0.5 * (b - a)[:, None] * np.tile(GL_W, (m, 1))
    cosang = r * r * np.cos(p * u) + rho2 * np.cos(q * u)
    cosang = np.clip(cosang, -1.0, 1.0)
    d = np.arccos(cosang)
    with np.errstate(divide="ignore"):
        f = 1.0 / (d * d) - 1.0 / (v2 * u * u)
    return 4.0 * np.pi * v2 * float((f * w).sum())


def minimize_r(p, q, fn, lo=1e-4, hi=1 - 1e-4, tol=1e-10):
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = fn(c), fn(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = fn(d)
    r = 0.5 * (a + b)
    return r, fn(r)


def minimize_r_log(p, q, fn, tol=1e-12):
    """Golden search in s = log(1-r): the E_d optimum approaches r=1 ~ q^-2."""
    g = (np.sqrt(5) - 1) / 2
    a, b = np.log(1e-9), np.log(0.5)
    c, d = b - g * (b - a), a + g * (b - a)
    F = lambda s: fn(1.0 - np.exp(s))
    fc, fd = F(c), F(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = F(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = F(d)
    s = 0.5 * (a + b)
    return 1.0 - np.exp(s), F(s)


def report(p, q):
    rd, Ed = minimize_r_log(p, q, lambda r: E_d(p, q, r))
    def Eq(r):
        v2 = p * p * r * r + q * q * (1 - r * r)
        return E_d(p, q, r) / (4 * np.pi ** 2 * v2)
    rq, Eqv = minimize_r(p, q, Eq)
    Lq = 2 * np.pi * np.sqrt(p * p * rq * rq + q * q * (1 - rq * rq))
    Ld = 2 * np.pi * np.sqrt(p * p * rd * rd + q * q * (1 - rd * rd))
    print(f"T({p},{q}): E_d_min={Ed:.6f} @ r={rd:.6f} (L={Ld:.4f})   "
          f"E_q_min={Eqv:.8f} @ r={rq:.6f} (L={Lq:.4f})")
    return dict(p=p, q=q, Ed=Ed, rd=rd, Ld=Ld, Eq=Eqv, rq=rq, Lq=Lq)


if __name__ == "__main__":
    specs = [tuple(int(x) for x in s.split(",")) for s in sys.argv[1:]]
    for p, q in specs:
        report(p, q)
