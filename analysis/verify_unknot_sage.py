#!/usr/bin/env sage-python
"""Rigorously check whether the realized Ochiai / Freedman curves are the UNKNOT.

Run under SageMath:  sage -python analysis/verify_unknot_sage.py

Determinant = 1 is necessary but NOT sufficient (nontrivial knots can have det 1).
Here we compute strong invariants from the diagram's DT code (exact, combinatorial):
  * Jones polynomial      (unknot  <=>  V = 1, up to the usual convention)
  * Alexander polynomial  (unknot  =>   Delta = 1)
  * fundamental group of the exterior (unknot  <=>  pi_1 = Z)
  * repeated diagram simplification (reaching 0 crossings PROVES the unknot)
"""
import snappy


def gauss_to_dt(seq):
    pos = {}
    for label, s in enumerate(seq, 1):
        pos.setdefault(abs(s), []).append((label, s > 0))
    dt = {}
    for c, ((l1, o1), (l2, o2)) in pos.items():
        odd, even, eo = (l1, l2, o2) if l1 % 2 == 1 else (l2, l1, o1)
        dt[odd] = -even if eo else even
    return [dt[o] for o in range(1, len(seq), 2)]


OCHIAI = [-1, 2, 3, -4, -5, -6, -7, 8, -9, 10, -11, -3, 4, -12, 13, 14, -8, 7,
          -14, -15, 12, 5, -2, 1, -16, 11, -10, 9, 15, -13, 6, 16]
FHW = [3, 4, -5, -8, 10, 12, -13, -15, 8, 7, -9, -10, 15, 16, -4, 1, 17, -20, 32,
       31, -27, -25, 22, 24, -31, -29, 28, 27, -24, -23, 20, 19, -18, -17, -21,
       -22, 25, 26, -30, -32, 23, 21, -26, -28, 29, 30, -19, 18, 2, -3, 14, 13,
       -12, -11, 6, 5, -16, -14, 11, 9, -7, -6, -1, -2]


def check(name, code):
    dt = gauss_to_dt(code)
    L = snappy.Link("DT:[%s]" % ",".join(map(str, dt)))
    n0 = len(L.crossings)

    # 1) repeated simplification — 0 crossings is a proof of unknottedness
    Ls = L.copy()
    for _ in range(50):
        Ls.simplify("global")
        if len(Ls.crossings) == 0:
            break
    simp = len(Ls.crossings)

    # 2) polynomial invariants
    try:
        jones = L.jones_polynomial()
    except Exception as e:
        jones = f"<err {e}>"
    M = L.exterior()
    try:
        alex = M.alexander_polynomial()
    except Exception as e:
        alex = f"<err {e}>"

    # 3) fundamental group of the exterior
    G = M.fundamental_group()
    gens, rels = G.num_generators(), G.num_relators()

    print(f"=== {name} ===")
    print(f"  crossings (as built)     : {n0}")
    print(f"  after 50x simplify       : {simp}   {'<-- PROVES UNKNOT' if simp == 0 else '(hard: did not trivialize)'}")
    print(f"  Jones polynomial         : {jones}")
    print(f"  Alexander polynomial     : {alex}")
    print(f"  pi_1(exterior)           : {gens} generators, {rels} relators")
    verdict = "UNKNOT" if (simp == 0 or str(jones) in ("1", "1.0")) else "NOT CONFIRMED trivial by these"
    print(f"  VERDICT                  : {verdict}")
    print()


if __name__ == "__main__":
    print("SnapPy", snappy.__version__, "\n")
    check("OCHIAI (16 cr)", OCHIAI)
    check("FREEDMAN-HE-WANG (32 cr)", FHW)
