#!/usr/bin/env python3
"""
E_q versus crossing number, c = 0 … 30, up to three knot types per c.

Design: at each crossing number take one representative from each of three
structurally different families, so that any trend in c is not a trend in
"family":

    torus       T(p,q),  c = min(p(q-1), q(p-1))      genus grows with c
    twist       m = c-2 half-twists (2-bridge)         hyperbolic, genus 1
    composite   connect sums, c additive               reducible

Not every c admits a torus knot (none for c = 4, 6, 12, 18, 30), and c ≤ 5 has
fewer than three knots in total, so those rows are short by construction.

Uniform protocol for every knot: quantity energy, same ITER/STEP, N scaled with
c, determinant checked on BOTH the generated start and the minimiser (a start
that builds the wrong type is discarded rather than minimised).

Stages:  python3 analysis/crossing_scan.py gen      # build + certify starts
         python3 analysis/crossing_scan.py min [-j4]# minimise
         python3 analysis/crossing_scan.py collect  # det + ACN + JSON
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = "/Users/yash/knot-s3"
OUT = os.path.join(ROOT, "output/_cscan")
PY = os.path.expanduser("~/.pyenv/shims/python3")
BIN = os.path.join(ROOT, "build/energy_s3")
ITER, STEP = 4000, 0.01


# ── determinants ───────────────────────────────────────────────────────────
def torus_det(p, q):
    """|Δ_{T(p,q)}(-1)|: 1 if p,q both odd, else the even one's partner."""
    if p % 2 and q % 2:
        return 1
    return q if p % 2 == 0 else p


def torus_c(p, q):
    return min(p * (q - 1), q * (p - 1))


# ── the knot list ──────────────────────────────────────────────────────────
def torus_by_c():
    """smallest-index torus knot at each crossing number"""
    best = {}
    for p in range(2, 9):
        for q in range(p + 1, 40):
            if np.gcd(p, q) != 1:
                continue
            c = torus_c(p, q)
            if c <= 30 and c not in best:
                best[c] = (p, q)
    return best


TORUS = torus_by_c()


def build_list():
    ks = [dict(tag="c00_unknot", name="unknot", family="unknot", c=0,
               gen=["torus", 1, 1], det=1, N=600)]
    for c in range(3, 31):
        # 1 — torus
        if c in TORUS:
            p, q = TORUS[c]
            n = 2 * q * q if p == 2 else 45 * c
            ks.append(dict(tag="c%02d_T%d_%d" % (c, p, q), name="T(%d,%d)" % (p, q),
                           family="torus", c=c, gen=["torus", p, q],
                           det=torus_det(p, q), N=int(np.clip(n, 1000, 2400))))
        # 2 — twist knot (m = c-2); m=1 is the trefoil, already the c=3 torus
        m = c - 2
        if m >= 2:
            ks.append(dict(tag="c%02d_tw%d" % (c, m), name="twist m=%d" % m,
                           family="twist", c=c, gen=["twist", m], det=2 * m + 1,
                           N=int(np.clip(70 * c, 1200, 2400))))
        # 3 — composite: even c -> 3_1 # T(2,c-3);  odd c -> 4_1 # T(2,c-4)
        comp = None
        if c >= 6 and (c - 3) % 2 == 1 and c - 3 >= 3:
            comp = dict(tag="c%02d_cs3_%d" % (c, c - 3),
                        name="3_1 # T(2,%d)" % (c - 3), gen=["connect", "2,3", "2,%d" % (c - 3)],
                        det=3 * (c - 3))
        elif c >= 7 and (c - 4) % 2 == 1 and c - 4 >= 3:
            comp = dict(tag="c%02d_csf8_%d" % (c, c - 4),
                        name="4_1 # T(2,%d)" % (c - 4), gen=["connect", "f8", "2,%d" % (c - 4)],
                        det=5 * (c - 4))
        if comp:
            comp.update(family="composite", c=c,
                        N=int(np.clip(70 * c, 1200, 2400)))
            ks.append(comp)
        # c with no torus knot: add a second composite so the row still has 3
        if c not in TORUS and c >= 6:
            if c == 6:
                # 3_1 # T(2,3) above IS the granny; the square (opposite
                # chirality, same det/genus/Alexander) is the third c=6 knot
                ks.append(dict(tag="c06_square", name="3_1 # 3_1* (square)",
                               family="composite", c=6, gen=["type", "square"],
                               det=9, N=1400))
            elif (c - 6) % 2 == 1 and c - 6 >= 3:
                ks.append(dict(tag="c%02d_cs33_%d" % (c, c - 6),
                               name="3_1 # 3_1 # T(2,%d)" % (c - 6), family="composite", c=c,
                               gen=["connect", "2,3", "2,3", "2,%d" % (c - 6)],
                               det=9 * (c - 6), N=int(np.clip(70 * c, 1400, 2400))))
    return ks


KNOTS = build_list()


# ── stages ─────────────────────────────────────────────────────────────────
def gen_cmd(k, path):
    g = k["gen"]
    if g[0] == "torus":
        return [PY, "knots/generate.py", str(g[1]), str(g[2]), "--n", str(k["N"]),
                "--out", os.path.dirname(path)]
    if g[0] == "twist":
        # generate_plat.py, NOT generate_twist.py: the latter builds unknots
        # (det 1 for every m — caught by the start-certification step)
        return [PY, "knots/generate_plat.py", "--twist", str(g[1]),
                "--n", str(k["N"]), "--out", path]
    if g[0] == "type":
        return [PY, "knots/generate_composites.py", "--type", g[1],
                "--n", str(k["N"]), "--out", path]
    return ([PY, "knots/generate_composites.py", "--connect"] + list(g[1:])
            + ["--n", str(k["N"]), "--out", path])


def det_of(path, tries=11):
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    from knot_check import robust_determinant, read_vect_pts  # noqa: E402
    try:
        d, votes, unan = robust_determinant(read_vect_pts(path), ntries=tries)
        return int(d), bool(unan)
    except Exception as e:                      # noqa: BLE001
        return None, str(e)


def stage_gen():
    os.makedirs(OUT, exist_ok=True)
    man = []
    for k in KNOTS:
        d = os.path.join(OUT, k["tag"])
        os.makedirs(d, exist_ok=True)
        init = os.path.join(d, "init.vect")
        if not os.path.exists(init):
            cmd = gen_cmd(k, init)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if k["gen"][0] == "torus":          # generate.py names its own file
                src = os.path.join(d, "T%d_%d.vect" % (k["gen"][1], k["gen"][2]))
                if os.path.exists(src):
                    os.replace(src, init)
            if not os.path.exists(init):
                k["gen_error"] = (r.stderr or r.stdout)[-300:]
                man.append(k); print("  FAIL gen %s" % k["tag"]); continue
        dd, un = det_of(init)
        k["det_start"], k["det_start_unanimous"] = dd, un
        k["ok_start"] = (dd == k["det"])
        print("  %-18s %-22s c=%2d N=%4d  det %s (want %s) %s"
              % (k["tag"], k["name"], k["c"], k["N"], dd, k["det"],
                 "OK" if k["ok_start"] else "MISMATCH"))
        man.append(k)
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    ok = sum(1 for k in man if k.get("ok_start"))
    print("\n%d/%d starts certified" % (ok, len(man)))


def min_one(k):
    d = os.path.join(OUT, k["tag"])
    log, fin = os.path.join(d, "log.csv"), os.path.join(d, "final.vect")
    if os.path.exists(fin) and os.path.getsize(fin) > 100:
        return k["tag"], "cached"
    cmd = [BIN, os.path.join(d, "init.vect"), log, fin, str(ITER), str(STEP),
           "--energy", "quantity"]
    # curvature-adaptive reparam starves plat/twist starts (turning is
    # concentrated at the clasp); composites need it against summand collapse
    cmd += ["--reparam", "0"] if k["family"] in ("torus", "twist", "unknot") \
        else ["--reparam", "50"]
    if k["family"] in ("torus", "unknot"):
        cmd += ["--no-normalize"]               # start is already on S³ exactly
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return k["tag"], ("ok" if os.path.exists(fin) else (r.stderr or "")[-200:])


def stage_min(jobs):
    man = json.load(open(os.path.join(OUT, "manifest.json")))
    todo = [k for k in man if k.get("ok_start")]
    print("minimising %d knots, %d at a time" % (len(todo), jobs))
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for i, (tag, st) in enumerate(ex.map(min_one, todo), 1):
            print("  [%2d/%d] %-18s %s" % (i, len(todo), tag, st), flush=True)


def acn(P, sub=700):
    """average crossing number, discrete Gauss double integral"""
    n = len(P)
    if n > sub:
        P = P[np.round(np.linspace(0, n, sub, endpoint=False)).astype(int)]
        n = len(P)
    T = np.roll(P, -1, 0) - P
    M = 0.5 * (P + np.roll(P, -1, 0))
    R = M[:, None, :] - M[None, :, :]
    d = np.linalg.norm(R, axis=-1)
    np.fill_diagonal(d, np.inf)
    cr = np.cross(np.broadcast_to(T[:, None, :], R.shape),
                  np.broadcast_to(T[None, :, :], R.shape))
    num = np.einsum("ijk,ijk->ij", cr, R)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.abs(num) / d ** 3
    return float(np.nansum(v) / (4 * np.pi))


def stage_collect():
    man = json.load(open(os.path.join(OUT, "manifest.json")))
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    from knot_check import read_vect_pts as read_vect            # noqa: E402
    rows = []
    for k in man:
        d = os.path.join(OUT, k["tag"])
        fin, log = os.path.join(d, "final.vect"), os.path.join(d, "log.csv")
        if not (os.path.exists(fin) and os.path.exists(log)):
            continue
        A = np.loadtxt(log, delimiter=",", skiprows=1, ndmin=2)
        if len(A) < 2:
            continue
        E, g, L = A[-1, 1], A[-1, 2], A[-1, 4]
        tail = A[max(0, len(A) - 200):, 1]
        k.update(E_q=float(E), gnorm=float(g), L=float(L), iters=int(A[-1, 0]),
                 tailrel=float((tail.max() - tail.min()) / abs(E)))
        dd, un = det_of(fin)
        k["det_final"], k["det_final_unanimous"] = dd, un
        k["type_held"] = (dd == k["det"])
        P = read_vect(fin)
        k["ACN"] = acn(P)
        k["ACN_over_c"] = k["ACN"] / k["c"] if k["c"] else None
        rows.append(k)
        print("  %-18s c=%2d  E_q=%.5f  L=%7.3f |g|=%.2e det %s %s"
              % (k["tag"], k["c"], E, L, g, dd, "HELD" if k["type_held"] else "LOST"))
    json.dump(rows, open(os.path.join(ROOT, "notes/crossing_scan_data.json"), "w"),
              indent=1)
    print("\n%d rows -> notes/crossing_scan_data.json" % len(rows))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if cmd == "gen":
        stage_gen()
    elif cmd == "min":
        j = int(sys.argv[2][2:]) if len(sys.argv) > 2 and sys.argv[2].startswith("-j") else 4
        stage_min(j)
    elif cmd == "collect":
        stage_collect()
    elif cmd == "list":
        for k in KNOTS:
            print("  c=%2d  %-24s %-10s det=%-4d N=%d"
                  % (k["c"], k["name"], k["family"], k["det"], k["N"]))
        print("\n%d knots" % len(KNOTS))
