/**
 * energy_s3.cpp
 *
 * Gradient flow for the S³ O'Hara energy  E^{(2)}_{S³}  of knots — the energy
 * of Conjecture 4.4 (existence of minimisers on S³ at exponent αp = 2):
 *
 *   E(K) = ∬_{K×K} ( 1/d_{S³}(x,y)²  −  1/d_K(x,y)² ) dx dy
 *
 *   d_{S³}(x,y) = arccos⟨x,y⟩   geodesic distance on S³ (NOT the Euclidean
 *                               chord — so E is NOT Möbius-invariant, exactly
 *                               as the paper notes).
 *   d_K(x,y)    = shorter arc length along the knot.
 *
 * Objective, gradient
 * -------------------
 *   compute_energy_ohara / compute_gradient_ohara implement E and its exact
 *   analytic ℝ⁴ gradient (validated against finite differences to ~1e-4, and
 *   against the paper's Clifford-torus benchmark min_r E ≈ 54.3 at r ≈ 0.86).
 *   The gradient is projected onto T_{x_k}S³.
 *
 * Sobolev preconditioning
 * -----------------------
 *   TangentPointMetric0::Solve (Repulsor) solves  A·g_sob = ∇E  via
 *   preconditioned GMRES.  A is used purely as a geometry-based smoother to
 *   make the step resolution-independent; the line search (below) enforces
 *   correctness regardless of the preconditioner.  This is the ONLY remaining
 *   use of Repulsor — its tangent-point energy is no longer the objective.
 *
 * Line search / convergence
 * --------------------------
 *   Backtracking Armijo on E^{(2)}_{S³} itself (strict decrease), so the
 *   logged energy is monotone.  Stops when no step decreases E, or progress
 *   stalls, or |∇E| → 0.
 *
 * Retraction
 * ----------
 *   x_k ← normalize( exp_{x_k}(−α · g_sob[k]) ),  k = 0…n−1.
 *
 * Compile:
 *   clang++ -std=c++20 -O3 -fenable-matrix -pthread \
 *     -I/Users/yash/knot-s3/Repulsor \
 *     -I/Users/yash/knot-s3/Repulsor/submodules/Tensors \
 *     -framework Accelerate \
 *     energy_s3.cpp -o energy_s3
 */

// ─── macOS Accelerate / OpenBLAS ───────────────────────────────────────────
#ifdef __APPLE__
#include "submodules/Tensors/Accelerate.hpp"
#else
#include "submodules/Tensors/OpenBLAS.hpp"
#endif

// ─── Repulsor ───────────────────────────────────────────────────────────────
#include "Repulsor.hpp"

// ─── STL ────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

using namespace Tools;

using Int = std::int32_t;
using LInt = std::int64_t;
using Real = double;

using Mesh_T = Repulsor::SimplicialMeshBase<Real, Int, LInt>;
using Energy_T = Repulsor::EnergyBase<Mesh_T>;
using Metric_T = Repulsor::MetricBase<Mesh_T>;

static constexpr int DOM_DIM = 1; // line segments
static constexpr int AMB_DIM = 4; // ambient ℝ⁴ (S³ ⊂ ℝ⁴)
static constexpr Real TPE_Q =
    3.0; // tangent-point exponent q  (matches "repel_curve 3 6")
static constexpr Real TPE_P = 6.0;    // tangent-point exponent p
static constexpr Real BH_THETA = 0.5; // Barnes-Hut opening angle

// ═══════════════════════════════════════════════════════════════════════════
// §1  S³ geometry utilities
// ═══════════════════════════════════════════════════════════════════════════

static constexpr Real EPS = 1e-14;
static bool CCD_ENABLED = false;  // toggled on via --ccd flag
static int G_THREADS = 1;         // set in main from hardware_concurrency
static int FRAME_EVERY = 0;       // --frames K: dump a trajectory frame every K
                                  // iterations for the live viewer (0 = off)
static std::string TRAJ_PATH;     // where the trajectory JSONL is written
static int REPARAM_EVERY = 50;    // --reparam K: re-uniformise arc length every K
                                  // iterations (0 = off; for collapse diagnosis)
static bool NORMALIZE = true;     // --no-normalize: skip the ℝ³ centre+RMS-scale
                                  // step before the S³ lift. That step is an ℝ³
                                  // similarity (NOT an S³ isometry), so it distorts
                                  // a knot already generated on S³ (torus knots);
                                  // skipping it starts T(p,q) on the pristine
                                  // Clifford torus.  Composites (generated in ℝ³ at
                                  // large scale) still want it ON.

inline Real dot4(const Real *a, const Real *b) noexcept {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
inline Real norm2_4(const Real *a) noexcept { return dot4(a, a); }
inline Real norm_4(const Real *a) noexcept { return std::sqrt(norm2_4(a)); }

inline void normalise4(Real *a) noexcept {
  Real n = norm_4(a);
  if (n > EPS) {
    a[0] /= n;
    a[1] /= n;
    a[2] /= n;
    a[3] /= n;
  }
}

/** Geodesic distance on S³: arccos(clamp(<x,y>,-1,1)). */
inline Real geodesic_dist(const Real *x, const Real *y) noexcept {
  Real d = std::max(-1.0, std::min(1.0, dot4(x, y)));
  return std::acos(d);
}

/**
 * Unit log-map from x toward y.  Stores unit direction in out[4].
 * Returns the geodesic distance θ = d(x,y).
 */
inline Real log_unit(const Real *x, const Real *y, Real *out) noexcept {
  Real c = std::max(-1.0, std::min(1.0, dot4(x, y)));
  Real theta = std::acos(c);
  // tangent component of y at x:  v = y - c*x
  Real v[4] = {y[0] - c * x[0], y[1] - c * x[1], y[2] - c * x[2],
               y[3] - c * x[3]};
  Real n = norm_4(v);
  if (n > EPS) {
    out[0] = v[0] / n;
    out[1] = v[1] / n;
    out[2] = v[2] / n;
    out[3] = v[3] / n;
  } else {
    out[0] = out[1] = out[2] = out[3] = 0.0;
  }
  return theta;
}

/** exp_x(v) — retraction on S³. */
inline void exp_map(const Real *x, const Real *v, Real *out) noexcept {
  Real nv = norm_4(v);
  if (nv < EPS) {
    out[0] = x[0];
    out[1] = x[1];
    out[2] = x[2];
    out[3] = x[3];
    return;
  }
  Real c = std::cos(nv), s = std::sin(nv);
  for (int k = 0; k < 4; ++k)
    out[k] = c * x[k] + s * v[k] / nv;
}

/** Orthogonal projection of v onto T_x S³. */
inline void proj_tangent(const Real *x, const Real *v, Real *out) noexcept {
  Real c = dot4(v, x);
  for (int k = 0; k < 4; ++k)
    out[k] = v[k] - c * x[k];
}

/**
 * Redistribute the n vertices to EQUAL geodesic arc length along the curve,
 * interpolating along the existing geodesic edges (slerp) so every new vertex
 * stays on S³ and the geometric curve is unchanged — only the parametrisation.
 *
 * The energy's gradient has a tangential component that slides vertices and
 * clusters them; left alone, a point-starved arc becomes too coarse to hold its
 * shape and a connect-sum summand collapses into a tiny coil. Re-parametrising
 * periodically keeps the resolution uniform so every part of the knot stays
 * intact.  Vertex 0 is held fixed as the arc-length origin.
 */
void reparametrize_s3(std::vector<Real> &v4, int n) {
  // Distribute the n vertices uniformly in the CURVATURE-ADAPTIVE measure
  //   dμ = ds + λ·dκ   (arc length + a turning term),  NOT pure arc length.
  //
  // Pure-arc-length reparametrisation hands a summand vertices in proportion to
  // its arc length, so a shrinking connect-sum summand is progressively starved
  // of points until it is too coarse to hold its crossings and unties — verified
  // empirically: arc-length reparam collapses T(2,3)#T(2,3) from det 9 to det 3
  // (a single trefoil, E 130→54), while disabling reparam preserves det 9. This
  // term fixes that without losing the anti-clustering benefit: a summand's total
  // turning stays ≈constant as it shrinks (its topology is fixed) while its arc
  // length →0, so the curvature term gives every summand a size-INDEPENDENT floor
  // of vertices (with CURV_WEIGHT=1, ≈25% per summand of a two-summand knot).
  // For a uniform-curvature curve (torus knot) the turning is spread evenly, so
  // the measure is ∝ arc length and this reduces to the old behaviour.
  constexpr Real CURV_WEIGHT = 1.0;   // curvature's share of the measure ≈ the floor

  std::vector<Real> L(n);             // geodesic edge lengths
  std::vector<Real> turn(n, 0.0);     // turning angle at each vertex
  Real total_arc = 0.0;
  for (int k = 0; k < n; ++k) {
    int kp = (k + 1) % n;
    Real c = std::max(-1.0, std::min(1.0, dot4(v4.data() + 4 * k,
                                               v4.data() + 4 * kp)));
    L[k] = std::acos(c);
    total_arc += L[k];
  }
  if (total_arc < EPS) return;

  // Turning angle at vertex k = angle between the ℝ⁴ chord edges e_{k-1}, e_k.
  Real total_turn = 0.0;
  for (int k = 0; k < n; ++k) {
    int km = (k - 1 + n) % n, kp = (k + 1) % n;
    Real e0[4], e1[4];
    for (int d = 0; d < 4; ++d) {
      e0[d] = v4[4 * k + d]  - v4[4 * km + d];
      e1[d] = v4[4 * kp + d] - v4[4 * k + d];
    }
    Real n0 = norm_4(e0), n1 = norm_4(e1);
    if (n0 > EPS && n1 > EPS) {
      Real cc = std::max(-1.0, std::min(1.0, dot4(e0, e1) / (n0 * n1)));
      turn[k] = std::acos(cc);
    }
    total_turn += turn[k];
  }
  Real lambda = (total_turn > EPS) ? CURV_WEIGHT * total_arc / total_turn : 0.0;

  // Per-edge weighted measure: arc length + half the turning at each endpoint.
  std::vector<Real> cumW(n + 1);
  cumW[0] = 0.0;
  for (int k = 0; k < n; ++k) {
    int kp = (k + 1) % n;
    cumW[k + 1] = cumW[k] + L[k] + lambda * 0.5 * (turn[k] + turn[kp]);
  }
  Real Wtot = cumW[n];
  if (Wtot < EPS) return;

  std::vector<Real> out(n * 4);
  int j = 0;
  for (int i = 0; i < n; ++i) {
    Real target = (Real)i * Wtot / n;
    while (j < n - 1 && cumW[j + 1] <= target) ++j;
    // Within an edge the measure is ≈uniform, so the weighted fraction is also
    // the geodesic arc-length fraction used by the slerp below.
    Real wj = cumW[j + 1] - cumW[j];
    Real t = (wj > EPS) ? (target - cumW[j]) / wj : 0.0;
    Real th = L[j];
    const Real *a = v4.data() + 4 * j;
    const Real *b = v4.data() + 4 * ((j + 1) % n);
    Real *o = out.data() + 4 * i;
    if (th < 1e-9) {
      for (int d = 0; d < 4; ++d) o[d] = a[d];
    } else {
      Real sn = std::sin(th);
      Real w0 = std::sin((1.0 - t) * th) / sn;
      Real w1 = std::sin(t * th) / sn;
      for (int d = 0; d < 4; ++d) o[d] = w0 * a[d] + w1 * b[d];
    }
    normalise4(o);
  }
  v4.swap(out);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2  Stereographic projection  ℝ³ ↔ S³
// ═══════════════════════════════════════════════════════════════════════════

inline void r3_to_s3(const Real *y, Real *x) noexcept {
  Real r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
  Real d = 1.0 / (r2 + 1.0);
  x[0] = 2.0 * y[0] * d;
  x[1] = 2.0 * y[1] * d;
  x[2] = 2.0 * y[2] * d;
  x[3] = (r2 - 1.0) * d;
}

inline void s3_to_r3(const Real *x, Real *y) {
  Real denom = 1.0 - x[3];
  if (std::fabs(denom) < EPS)
    throw std::runtime_error("s3_to_r3: near north pole");
  y[0] = x[0] / denom;
  y[1] = x[1] / denom;
  y[2] = x[2] / denom;
}

// ═══════════════════════════════════════════════════════════════════════════
// §3  File I/O
// ═══════════════════════════════════════════════════════════════════════════

std::vector<Real> read_vect(const std::string &path, int &n_out) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("Cannot open: " + path);
  int nc;
  f >> nc;
  if (nc != 1)
    std::cerr << "Warning: " << nc << " components; using first only.\n";
  int n;
  f >> n;
  std::vector<Real> pts(n * 3);
  for (int i = 0; i < n; ++i)
    f >> pts[3 * i] >> pts[3 * i + 1] >> pts[3 * i + 2];
  n_out = n;
  return pts;
}

void write_vect(const std::string &path, const std::vector<Real> &pts_r3,
                int n) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("Cannot write: " + path);
  f << "1\n" << n << "\n" << std::setprecision(10) << std::fixed;
  for (int i = 0; i < n; ++i)
    f << pts_r3[3 * i] << " " << pts_r3[3 * i + 1] << " " << pts_r3[3 * i + 2]
      << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  S³ O'Hara energy  E^{(2)}_{S³}   (the conjecture's energy)
//
//   E(K) = ∬_{K×K} ( 1/d_{S³}(x,y)²  −  1/d_K(x,y)² ) dx dy
//
//   d_{S³}(x,y) = arccos⟨x,y⟩            geodesic distance on S³
//   d_K(x,y)    = shorter arc length along the knot
//   dx, dy      = arc-length (Voronoi) measure
//
//   Discrete form (matches the validated Python reference and reproduces the
//   paper's Clifford-torus benchmark  min_r E ≈ 54.3 at r ≈ 0.86):
//
//     E = 2 · Σ_{i<j, non-adjacent} ( 1/θ_ij² − 1/a_ij² ) · w_i w_j
//
//   θ_ij = arccos⟨x_i,x_j⟩,  L_k = arccos⟨x_k,x_{k+1}⟩,  T = Σ L_k,
//   fwd_ij = Σ_{i≤k<j} L_k,  a_ij = min(fwd_ij, T − fwd_ij),
//   w_i = (L_{i−1}+L_i)/2.  Cyclically-adjacent pairs have θ = a = L, so their
//   integrand is identically 0 (and their gradient cancels) — skipped.  The
//   leading 2 is the ordered double integral over K×K.
// ═══════════════════════════════════════════════════════════════════════════

// Edge geodesic lengths L, prefix sums S (size n+1, S[n]=T), Voronoi weights w.
static void ohara_geometry(const std::vector<Real> &v4, int n,
                           std::vector<Real> &L, std::vector<Real> &S,
                           std::vector<Real> &w, Real &T) {
  L.resize(n);
  S.resize(n + 1);
  w.resize(n);
  S[0] = 0.0;
  for (int k = 0; k < n; ++k) {
    int kp = (k + 1) % n;
    Real c = std::max(-1.0, std::min(1.0, dot4(v4.data() + 4 * k,
                                               v4.data() + 4 * kp)));
    L[k] = std::acos(c);
    S[k + 1] = S[k] + L[k];
  }
  T = S[n];
  for (int k = 0; k < n; ++k)
    w[k] = 0.5 * (L[(k - 1 + n) % n] + L[k]);
}

Real compute_energy_ohara(const std::vector<Real> &v4, int n) {
  std::vector<Real> L, S, w;
  Real T;
  ohara_geometry(v4, n, L, S, w, T);

  const int nt = std::clamp((n * n) / 200000, 1, G_THREADS);
  std::vector<Real> partial(nt, 0.0);

  auto worker = [&](int tid) {
    Real acc = 0.0;
    for (int i = tid; i < n; i += nt) {
      const Real *xi = v4.data() + 4 * i;
      const int j_end = (i == 0) ? n - 1 : n; // (0,n-1) cyclically adjacent
      for (int j = i + 2; j < j_end; ++j) {
        const Real *xj = v4.data() + 4 * j;
        Real c = std::max(-1.0, std::min(1.0, dot4(xi, xj)));
        Real theta = std::acos(c);
        if (theta < EPS)
          continue; // coincident: integrable singularity, drop the term
        Real fwd = S[j] - S[i];
        Real a = std::min(fwd, T - fwd);
        acc += (1.0 / (theta * theta) - 1.0 / (a * a)) * w[i] * w[j];
      }
    }
    partial[tid] = acc;
  };

  if (nt == 1) {
    worker(0);
  } else {
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (int t = 0; t < nt; ++t)
      pool.emplace_back(worker, t);
    for (auto &th : pool)
      th.join();
  }

  Real E = 0.0;
  for (int t = 0; t < nt; ++t)
    E += partial[t];
  return 2.0 * E; // ordered double integral over K×K
}

// ═══════════════════════════════════════════════════════════════════════════
// §4a  Analytic ℝ⁴ gradient of E^{(2)}_{S³}
//
// Differentiates the discrete energy above exactly.  Per non-adjacent pair
// (i<j) with G = 1/θ² − 1/a² and contribution 2·G·w_i w_j to E:
//
//   • θ-part  (θ = arccos⟨x_i,x_j⟩): direct, accumulates into grad[i], grad[j].
//   • a-part  (a = Σ L_k over the shorter arc): adds a constant β to every
//     edge on that arc.  Done in O(1)/pair via a difference array, prefix-
//     summed once at the end ⇒ the whole gradient stays O(n²), not O(n³).
//   • w-part  (w depends on incident edge lengths): accumulates into W_i, W_j.
//
//   The arc and weight contributions are both linear in the edge lengths L_k,
//   so they are merged into one per-edge coefficient Γ_k and applied through
//   ∂L_k/∂{x_k,x_{k+1}} in a single O(n) sweep.  Caller projects onto T_xS³.
// ═══════════════════════════════════════════════════════════════════════════

void compute_gradient_ohara(const std::vector<Real> &v4, int n, Real *grad) {
  std::vector<Real> L, S, w;
  Real T;
  ohara_geometry(v4, n, L, S, w, T);

  const int nt = std::clamp((n * n) / 200000, 1, G_THREADS);
  // Per-thread accumulators: gradient, W (∂E/∂w_i), arc difference array.
  std::vector<std::vector<Real>> gbuf(nt, std::vector<Real>(n * 4, 0.0));
  std::vector<std::vector<Real>> Wbuf(nt, std::vector<Real>(n, 0.0));
  std::vector<std::vector<Real>> Dbuf(nt, std::vector<Real>(n + 1, 0.0));

  auto worker = [&](int tid) {
    Real *g = gbuf[tid].data();
    Real *W = Wbuf[tid].data();
    Real *D = Dbuf[tid].data();
    for (int i = tid; i < n; i += nt) {
      const Real *xi = v4.data() + 4 * i;
      const int j_end = (i == 0) ? n - 1 : n;
      for (int j = i + 2; j < j_end; ++j) {
        const Real *xj = v4.data() + 4 * j;
        Real c = std::max(-1.0, std::min(1.0, dot4(xi, xj)));
        Real theta = std::acos(c);
        if (theta < EPS)
          continue;
        Real fwd = S[j] - S[i];
        bool fwd_short = (fwd <= T - fwd);
        Real a = fwd_short ? fwd : T - fwd;
        Real G = 1.0 / (theta * theta) - 1.0 / (a * a);

        // θ-part:  ∂(2 w_i w_j / θ²)  with ∂θ/∂x_i = −x_j/sinθ
        Real sin_t = std::sqrt(std::max(0.0, 1.0 - c * c));
        if (sin_t > EPS) {
          Real kth = 4.0 * w[i] * w[j] / (theta * theta * theta * sin_t);
          for (int d = 0; d < 4; ++d) {
            g[4 * i + d] += kth * xj[d];
            g[4 * j + d] += kth * xi[d];
          }
        }

        // w-part accumulators:  ∂E/∂w_i gets 2 G w_j (and symmetric)
        W[i] += 2.0 * G * w[j];
        W[j] += 2.0 * G * w[i];

        // a-part:  β = ∂(−2 w_i w_j / a²)/∂a · (∂a/∂L = 1 on the shorter arc)
        Real beta = 4.0 * w[i] * w[j] / (a * a * a);
        if (fwd_short) {
          // shorter arc = forward edges [i, j-1]
          D[i] += beta;
          D[j] -= beta;
        } else {
          // shorter arc = backward edges [j, n-1] ∪ [0, i-1]
          D[j] += beta;
          D[n] -= beta;
          if (i > 0) {
            D[0] += beta;
            D[i] -= beta;
          }
        }
      }
    }
  };

  if (nt == 1) {
    worker(0);
  } else {
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (int t = 0; t < nt; ++t)
      pool.emplace_back(worker, t);
    for (auto &th : pool)
      th.join();
  }

  // Reduce per-thread buffers.
  for (int i = 0; i < n * 4; ++i)
    grad[i] = 0.0;
  std::vector<Real> W(n, 0.0), Dsum(n + 1, 0.0);
  for (int t = 0; t < nt; ++t) {
    for (int i = 0; i < n * 4; ++i)
      grad[i] += gbuf[t][i];
    for (int i = 0; i < n; ++i)
      W[i] += Wbuf[t][i];
    for (int i = 0; i <= n; ++i)
      Dsum[i] += Dbuf[t][i];
  }

  // Prefix-sum the difference array → per-edge arc coefficient.
  std::vector<Real> arc_coef(n, 0.0);
  Real run = 0.0;
  for (int k = 0; k < n; ++k) {
    run += Dsum[k];
    arc_coef[k] = run;
  }

  // Merge arc + weight contributions into one per-edge coefficient and apply
  // through  L_k = arccos⟨x_k,x_{k+1}⟩,  ∇_{x_k}L_k = −x_{k+1}/sin L_k.
  for (int k = 0; k < n; ++k) {
    int kp = (k + 1) % n;
    Real sin_L = std::sin(L[k]);
    if (sin_L <= EPS)
      continue;
    Real Gamma = arc_coef[k] + 0.5 * (W[k] + W[kp]);
    Real coef = -Gamma / sin_L;
    const Real *xk = v4.data() + 4 * k;
    const Real *xp = v4.data() + 4 * kp;
    for (int d = 0; d < 4; ++d) {
      grad[4 * k + d] += coef * xp[d];
      grad[4 * kp + d] += coef * xk[d];
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// §4b  CCD: Minimum edge-edge distance guard
//      Prevents topological tunneling in composite knots by rejecting
//      any trial step where non-adjacent edges come closer than d_min.
// ═══════════════════════════════════════════════════════════════════════════

/** Closest point on segment [A,B] to point P in ℝ⁴. Returns parameter t. */
inline Real closest_point_on_seg(const Real *A, const Real *B, const Real *P) {
  Real AB[4], AP[4];
  for (int d = 0; d < 4; ++d) { AB[d] = B[d]-A[d]; AP[d] = P[d]-A[d]; }
  Real ab2 = dot4(AB, AB);
  if (ab2 < EPS) return 0.0;
  Real t = dot4(AP, AB) / ab2;
  return std::max(0.0, std::min(1.0, t));
}

/** Squared chord distance between segments [A,B] and [C,D] in ℝ⁴. */
inline Real seg_seg_dist2(const Real *A, const Real *B,
                         const Real *C, const Real *D) {
  // Iterative closest-point approach (2 iterations is sufficient)
  Real s = 0.5, t = 0.5;
  for (int iter = 0; iter < 3; ++iter) {
    // Point on seg2 at parameter t
    Real Q[4];
    for (int d = 0; d < 4; ++d) Q[d] = C[d] + t*(D[d]-C[d]);
    // Closest point on seg1 to Q
    s = closest_point_on_seg(A, B, Q);
    // Point on seg1 at parameter s
    Real P[4];
    for (int d = 0; d < 4; ++d) P[d] = A[d] + s*(B[d]-A[d]);
    // Closest point on seg2 to P
    t = closest_point_on_seg(C, D, P);
  }
  Real P[4], Q[4], diff[4];
  for (int d = 0; d < 4; ++d) {
    P[d] = A[d] + s*(B[d]-A[d]);
    Q[d] = C[d] + t*(D[d]-C[d]);
    diff[d] = P[d] - Q[d];
  }
  return dot4(diff, diff);
}

// ═══════════════════════════════════════════════════════════════════════════
// §4c  Continuous collision detection (CCD) in ℝ³
//
// Knotting lives in the stereographic ℝ³ image, not in the ℝ⁴ chord metric, so
// the only rigorous self-avoidance test projects to ℝ³ and asks: as the curve
// moves linearly from its old to its trial position, do any two non-adjacent
// edges PASS THROUGH each other?  We reject the trial step iff some edge pair
// has a time-of-impact t∈(0,1].  This forbids the crossing event itself, so it
// is tunnel-proof at any step size and never freezes (arbitrarily close
// approaches are fine as long as nothing passes through).
// ═══════════════════════════════════════════════════════════════════════════

inline void sub3(const Real *a, const Real *b, Real *o) {
  o[0] = a[0]-b[0]; o[1] = a[1]-b[1]; o[2] = a[2]-b[2];
}
inline void cross3(const Real *a, const Real *b, Real *o) {
  o[0] = a[1]*b[2]-a[2]*b[1]; o[1] = a[2]*b[0]-a[0]*b[2]; o[2] = a[0]*b[1]-a[1]*b[0];
}
inline Real dot3(const Real *a, const Real *b) {
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

// Min distance² between segments [P,P+u] and [Q,Q+v] in ℝ³ (clamped params).
inline Real seg_seg_dist2_r3(const Real *P, const Real *u,
                             const Real *Q, const Real *v) {
  Real w0[3]; sub3(P, Q, w0);
  Real a = dot3(u, u), b = dot3(u, v), c = dot3(v, v);
  Real d = dot3(u, w0), e = dot3(v, w0);
  Real D = a*c - b*b, s, t;
  if (D < 1e-12) { s = 0.0; t = (c > 1e-12) ? e/c : 0.0; }
  else { s = (b*e - c*d)/D; t = (a*e - b*d)/D; }
  s = std::max(0.0, std::min(1.0, s));
  t = std::max(0.0, std::min(1.0, t));
  Real dd = 0.0;
  for (int k = 0; k < 3; ++k) {
    Real diff = w0[k] + s*u[k] - t*v[k];
    dd += diff*diff;
  }
  return dd;
}

// Do moving edges (A,B) and (C,D) — each vertex linearly interpolated from its
// "0" to its "1" position — cross in ℝ³ during t∈(0,1]?  Coplanarity is a cubic
// in t; we sample for sign changes, bisect to each root, and confirm the
// segments actually meet there (distance below `tol`).
bool edge_edge_ccd(const Real *A0, const Real *B0, const Real *C0, const Real *D0,
                   const Real *A1, const Real *B1, const Real *C1, const Real *D1,
                   Real tol) {
  auto cop = [&](Real t) {
    Real A[3], B[3], C[3], Dp[3], e1[3], e2[3], e3[3], cr[3];
    for (int k = 0; k < 3; ++k) {
      A[k]  = A0[k] + t*(A1[k]-A0[k]);
      B[k]  = B0[k] + t*(B1[k]-B0[k]);
      C[k]  = C0[k] + t*(C1[k]-C0[k]);
      Dp[k] = D0[k] + t*(D1[k]-D0[k]);
    }
    sub3(B, A, e1); sub3(Dp, C, e2); sub3(C, A, e3);
    cross3(e1, e2, cr);
    return dot3(cr, e3);
  };
  auto crosses_at = [&](Real t) {
    Real A[3], u[3], C[3], v[3], B[3], Dp[3];
    for (int k = 0; k < 3; ++k) {
      A[k]  = A0[k] + t*(A1[k]-A0[k]);
      B[k]  = B0[k] + t*(B1[k]-B0[k]);
      C[k]  = C0[k] + t*(C1[k]-C0[k]);
      Dp[k] = D0[k] + t*(D1[k]-D0[k]);
    }
    sub3(B, A, u); sub3(Dp, C, v);
    return seg_seg_dist2_r3(A, u, C, v) < tol*tol;
  };
  const int S = 8;
  Real tp = 0.0, fp = cop(0.0);
  for (int s = 1; s <= S; ++s) {
    Real t = (Real)s / S, ft = cop(t);
    if (fp == 0.0 || fp*ft < 0.0) {
      Real lo = tp, hi = t, flo = fp;
      for (int b = 0; b < 24; ++b) {
        Real mid = 0.5*(lo+hi), fm = cop(mid);
        if (flo*fm <= 0.0) hi = mid; else { lo = mid; flo = fm; }
      }
      Real tr = 0.5*(lo+hi);
      if (tr > 1e-9 && crosses_at(tr)) return true;
    }
    tp = t; fp = ft;
  }
  return false;
}

// Pick a unit ℝ⁴ direction `p` that stays far (geodesically) from EVERY vertex
// of both configurations, so the stereographic projection used by the CCD below
// is well-conditioned. Projecting from the fixed north pole (0,0,0,1) is the bug
// that let connect-sums tunnel: whenever a strand wandered near that pole its ℝ³
// image flew off toward infinity, the linear y0→y1 interpolation no longer
// tracked the true S³ motion, and the displacement-budget prune under-counted
// the motion — so a genuine self-passage near the pole slipped through. We
// instead choose the pole in the curve's emptiest direction (minimise
// max_i⟨p,x_i⟩ over a fixed candidate set; the curve is 1-D in S³ so almost
// every direction is empty), guaranteeing 1−⟨p,x_i⟩ is bounded away from 0.
static void choose_far_pole(const std::vector<Real> &v4a,
                            const std::vector<Real> &v4b, int n, Real *p) {
  static const Real cand[][4] = {
      {1, 0, 0, 0},    {-1, 0, 0, 0},   {0, 1, 0, 0},    {0, -1, 0, 0},
      {0, 0, 1, 0},    {0, 0, -1, 0},   {0, 0, 0, 1},    {0, 0, 0, -1},
      {0.5, 0.5, 0.5, 0.5},   {-0.5, 0.5, 0.5, 0.5},  {0.5, -0.5, 0.5, 0.5},
      {0.5, 0.5, -0.5, 0.5},  {0.5, 0.5, 0.5, -0.5},  {-0.5, -0.5, 0.5, 0.5},
      {0.5, -0.5, -0.5, 0.5}, {-0.5, 0.5, -0.5, 0.5},
  };
  const int NC = (int)(sizeof(cand) / sizeof(cand[0]));
  Real best = 1e30;
  int best_k = 0;
  for (int k = 0; k < NC; ++k) {
    Real worst = -1e30;
    for (int i = 0; i < n; ++i) {
      worst = std::max(worst, std::max(dot4(cand[k], v4a.data() + 4 * i),
                                       dot4(cand[k], v4b.data() + 4 * i)));
    }
    if (worst < best) { best = worst; best_k = k; }
  }
  for (int d = 0; d < 4; ++d) p[d] = cand[best_k][d];
}

// True iff any non-adjacent edge pair crosses while the curve moves from v4_old
// to v4_new (both S³ vertices; compared in the ℝ³ stereographic projection).
bool any_self_passage(const std::vector<Real> &v4_old,
                      const std::vector<Real> &v4_new, int n) {
  // Project both configurations to ℝ³ from a pole chosen far from the curve.
  // The Householder reflection H (= I − 2 vvᵀ/|v|², v = p − e₄) maps the chosen
  // pole p ↦ e₄ = (0,0,0,1); reflections preserve crossings, so projecting the
  // H-rotated coordinates from e₄ is equivalent to projecting the originals from
  // p, but reuses the standard formula. denom = 1 − (Hx)₃ = 1 − ⟨p,x⟩ is then
  // bounded away from 0 for every vertex (no pole blow-up).
  Real p[4];
  choose_far_pole(v4_old, v4_new, n, p);
  Real vv[4] = {p[0], p[1], p[2], p[3] - 1.0};
  Real vn2 = dot4(vv, vv);
  std::vector<Real> y0(n*3), y1(n*3);
  auto proj = [&](const std::vector<Real> &v4, std::vector<Real> &y) {
    for (int i = 0; i < n; ++i) {
      const Real *x = v4.data() + 4*i;
      Real hx[4];
      if (vn2 > 1e-12) {
        Real c = 2.0 * dot4(vv, x) / vn2;
        for (int d = 0; d < 4; ++d) hx[d] = x[d] - c * vv[d];
      } else {
        for (int d = 0; d < 4; ++d) hx[d] = x[d];
      }
      Real denom = 1.0 - hx[3];
      if (std::fabs(denom) < 1e-6) denom = (denom < 0 ? -1e-6 : 1e-6);
      y[3*i+0] = hx[0]/denom; y[3*i+1] = hx[1]/denom; y[3*i+2] = hx[2]/denom;
    }
  };
  proj(v4_old, y0); proj(v4_new, y1);

  // Per-vertex ℝ³ displacement, and a tolerance scaled to the mean edge.
  Real mean_edge = 0.0;
  for (int i = 0; i < n; ++i) {
    int ip = (i+1)%n;
    Real e[3]; sub3(y0.data()+3*ip, y0.data()+3*i, e);
    mean_edge += std::sqrt(dot3(e,e));
  }
  mean_edge /= n;
  const Real tol = 1e-4 * mean_edge;
  std::vector<Real> disp(n);
  for (int i = 0; i < n; ++i) {
    Real d[3]; sub3(y1.data()+3*i, y0.data()+3*i, d);
    disp[i] = std::sqrt(dot3(d,d));
  }

  const int SKIP = 3;
  const int nt = std::clamp((n*n)/200000, 1, G_THREADS);
  std::atomic<bool> hit{false};
  auto worker = [&](int tid) {
    for (int i = tid; i < n && !hit.load(std::memory_order_relaxed); i += nt) {
      int ip = (i+1)%n;
      for (int j = i+SKIP; j < n; ++j) {
        int jp = (j+1)%n;
        if (j-i < SKIP || n-(j-i) < SKIP) continue;
        if (i == 0 && jp == 0) continue;
        // Prune: if the two edges' closest static distance exceeds the total
        // motion of all four endpoints, they cannot pass through this step.
        Real ui[3], vj[3];
        sub3(y0.data()+3*ip, y0.data()+3*i, ui);
        sub3(y0.data()+3*jp, y0.data()+3*j, vj);
        Real d2 = seg_seg_dist2_r3(y0.data()+3*i, ui, y0.data()+3*j, vj);
        Real budget = disp[i]+disp[ip]+disp[j]+disp[jp];
        if (std::sqrt(d2) > budget) continue;
        if (edge_edge_ccd(y0.data()+3*i, y0.data()+3*ip, y0.data()+3*j, y0.data()+3*jp,
                          y1.data()+3*i, y1.data()+3*ip, y1.data()+3*j, y1.data()+3*jp,
                          tol)) {
          hit.store(true, std::memory_order_relaxed);
          return;
        }
      }
    }
  };
  if (nt == 1) worker(0);
  else {
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t) pool.emplace_back(worker, t);
    for (auto &th : pool) th.join();
  }
  return hit.load();
}

/**
 * True global minimum non-adjacent edge-edge distance (chord, ℝ⁴).
 * Unlike min_nonadjacent_edge_dist this never fast-exits — it returns the
 * actual minimum, which is what the tunnel-proof step cap needs. Threaded,
 * pruned by each thread's running minimum.
 */
Real global_min_gap(const std::vector<Real> &v4, int n) {
  const int SKIP = 3;
  const int nt = std::clamp((n * n) / 200000, 1, G_THREADS);
  std::vector<Real> local_min(nt, 1e30);

  auto worker = [&](int tid) {
    Real min_d2 = 1e30;
    for (int i = tid; i < n; i += nt) {
      int ip = (i + 1) % n;
      const Real *A = v4.data() + 4 * i;
      const Real *B = v4.data() + 4 * ip;
      Real ab[4];
      for (int d = 0; d < 4; ++d) ab[d] = B[d] - A[d];
      Real len1 = norm_4(ab);
      for (int j = i + SKIP; j < n; ++j) {
        int jp = (j + 1) % n;
        int diff_ji = j - i;
        int diff_jpi = (jp > i) ? (jp - i) : (n - i + jp);
        if (diff_ji < SKIP || (n - diff_ji) < SKIP) continue;
        if (diff_jpi < SKIP || (n - diff_jpi) < SKIP) continue;
        const Real *C = v4.data() + 4 * j;
        const Real *D = v4.data() + 4 * jp;
        Real cd[4];
        for (int d = 0; d < 4; ++d) cd[d] = D[d] - C[d];
        Real len2 = norm_4(cd);
        // Prune: closest the two segments can get is |A-C| - len1 - len2.
        // If that lower bound already exceeds the running min, skip.
        Real pt_d2 = 0.0;
        for (int d = 0; d < 4; ++d) { Real e = A[d] - C[d]; pt_d2 += e * e; }
        Real lb = std::sqrt(pt_d2) - len1 - len2;
        if (lb > 0 && lb * lb > min_d2) continue;
        Real d2 = seg_seg_dist2(A, B, C, D);
        if (d2 < min_d2) min_d2 = d2;
      }
    }
    local_min[tid] = min_d2;
  };

  if (nt == 1) {
    worker(0);
  } else {
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (int t = 0; t < nt; ++t) pool.emplace_back(worker, t);
    for (auto &th : pool) th.join();
  }
  Real m = 1e30;
  for (int t = 0; t < nt; ++t) m = std::min(m, local_min[t]);
  return std::sqrt(m);
}

/**
 * Check if any non-adjacent edges are closer than d_min (chord distance).
 * Returns the minimum distance found. O(N²) but optimized with a fast-path
 * point distance check to skip the expensive segment math.
 */
Real min_nonadjacent_edge_dist(const std::vector<Real> &v4, int n, Real d_safe, Real avg_edge_len) {
  Real d_safe2 = d_safe * d_safe;

  const int SKIP = 3; // skip edges within ±3 in the polygon
  const int nt = std::clamp((n * n) / 200000, 1, G_THREADS);
  std::vector<Real> local_min(nt, 1e30);
  std::atomic<bool> collision{false}; // lets all threads bail on first hit

  auto worker = [&](int tid) {
    Real min_d2 = 1e30;
    for (int i = tid; i < n; i += nt) {
      if (collision.load(std::memory_order_relaxed))
        break;
      int ip = (i+1) % n;
      const Real *A = v4.data() + 4*i;
      const Real *B = v4.data() + 4*ip;

      // Calculate exact length of segment 1
      Real diff_AB[4];
      for (int d = 0; d < 4; ++d) diff_AB[d] = B[d] - A[d];
      Real len1 = norm_4(diff_AB);

      for (int j = i+SKIP; j < n; ++j) {
        // Also skip if j+1 wraps around to be adjacent to i
        int jp = (j+1) % n;
        int diff_ji = j - i;
        int diff_jpi = (jp > i) ? (jp - i) : (n - i + jp);
        if (diff_ji < SKIP || (n - diff_ji) < SKIP) continue;
        if (diff_jpi < SKIP || (n - diff_jpi) < SKIP) continue;

        const Real *C = v4.data() + 4*j;
        const Real *D = v4.data() + 4*jp;

        // Calculate exact length of segment 2
        Real diff_CD[4];
        for (int d = 0; d < 4; ++d) diff_CD[d] = D[d] - C[d];
        Real len2 = norm_4(diff_CD);

        // Fast-path bounding sphere check:
        // If start points are further than (d_safe + length1 + length2), they CANNOT intersect.
        Real max_pt_dist = d_safe + len1 + len2;
        Real max_pt_dist2 = max_pt_dist * max_pt_dist;

        Real pt_d2 = 0.0;
        for (int d = 0; d < 4; ++d) {
          Real diff = A[d] - C[d];
          pt_d2 += diff * diff;
        }

        if (pt_d2 > max_pt_dist2) continue; // Skip expensive seg-seg math!

        Real d2 = seg_seg_dist2(A, B, C, D);
        if (d2 < min_d2) min_d2 = d2;
        if (d2 < d_safe2) { // FAST EXIT on first collision found
          collision.store(true, std::memory_order_relaxed);
          local_min[tid] = min_d2;
          return;
        }
      }
    }
    local_min[tid] = min_d2;
  };

  if (nt == 1) {
    worker(0);
  } else {
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (int t = 0; t < nt; ++t)
      pool.emplace_back(worker, t);
    for (auto &th : pool)
      th.join();
  }

  Real min_d2 = 1e30;
  for (int t = 0; t < nt; ++t)
    min_d2 = std::min(min_d2, local_min[t]);
  return std::sqrt(min_d2);
}

// ═══════════════════════════════════════════════════════════════════════════
// §5  Repulsor mesh construction
// ═══════════════════════════════════════════════════════════════════════════

std::unique_ptr<Mesh_T> build_mesh(
    Repulsor::SimplicialMesh_Factory<Mesh_T, DOM_DIM, DOM_DIM, AMB_DIM, AMB_DIM>
        &fac,
    const std::vector<Real> &v4, int n, int threads = 1) {
  std::vector<Int> simplices(n * 2);
  for (int i = 0; i < n; ++i) {
    simplices[2 * i] = i;
    simplices[2 * i + 1] = (i + 1) % n;
  }

  auto M = fac.Make(v4.data(), n, AMB_DIM, false, simplices.data(), n,
                    DOM_DIM + 1, false, threads);
  auto &m = *M;
  m.cluster_tree_settings.split_threshold = 2;
  m.block_cluster_tree_settings.far_field_separation_parameter = BH_THETA;
  m.adaptivity_settings.theta = 10.0;
  return M;
}

// ═══════════════════════════════════════════════════════════════════════════
// §6  Gradient descent on S³
// ═══════════════════════════════════════════════════════════════════════════

// Append one trajectory frame (iteration, energy, R³ stereographic coords) as a
// single JSON line.  The file is flushed each frame so a live viewer tailing it
// sees new frames as soon as they are produced.
static void write_frame(std::ofstream &f, const std::vector<Real> &v4, int n,
                        int iter, Real energy) {
  f << "{\"i\":" << iter << ",\"e\":" << std::setprecision(8) << energy
    << ",\"p\":[";
  for (int i = 0; i < n; ++i) {
    const Real *x = v4.data() + 4 * i;
    Real denom = 1.0 - x[3];
    if (std::fabs(denom) < 1e-6) denom = (denom < 0 ? -1e-6 : 1e-6);
    for (int d = 0; d < 3; ++d) {
      if (i || d) f << ',';
      f << std::setprecision(5) << x[d] / denom;
    }
  }
  f << "]}\n";
  f.flush();
}

void gradient_descent(std::vector<Real> &v4, // in/out: S³ vertices [n×4]
                      int n, const std::string &log_path, int max_iter,
                      Real alpha0) {
  // ── Repulsor factories ────────────────────────────────────────────────
  Repulsor::SimplicialMesh_Factory<Mesh_T, DOM_DIM, DOM_DIM, AMB_DIM, AMB_DIM>
      mesh_fac;

  // Energy: AllPairs for n<50, Barnes-Hut (TangentPointEnergy0) for n≥50
  Repulsor::TangentPointEnergy_AllPairs_Factory<Mesh_T, DOM_DIM, DOM_DIM,
                                                AMB_DIM, AMB_DIM>
      tpe_ap_fac;
  Repulsor::TangentPointEnergy0_Factory<Mesh_T, DOM_DIM, DOM_DIM, AMB_DIM,
                                        AMB_DIM>
      tpe_bh_fac;

  // Sobolev H^{1/2} metric
  Repulsor::TangentPointMetric0_Factory<Mesh_T, DOM_DIM, DOM_DIM, AMB_DIM,
                                        AMB_DIM>
      tpm_fac;

  auto tpe_ap_ptr = tpe_ap_fac.Make(DOM_DIM, AMB_DIM, TPE_Q, TPE_P);
  auto tpe_bh_ptr = tpe_bh_fac.Make(DOM_DIM, AMB_DIM, TPE_Q, TPE_P);
  auto tpm_ptr = tpm_fac.Make(DOM_DIM, AMB_DIM, TPE_Q, TPE_P);

  auto &E_ap = *tpe_ap_ptr; // AllPairs energy (small n)
  auto &E_bh = *tpe_bh_ptr; // BH energy (large n)
  auto &metric = *tpm_ptr;  // Sobolev H^{1/2} metric

  // Repulsor parallelises via ParallelDo, which creates and joins OS threads
  // on EVERY parallel section (each sparse mat-vec of the GMRES solve, etc.).
  // For curve meshes the per-call work is microseconds, so thread churn
  // dominates wall time. Cap Repulsor's thread count by problem size; the
  // O(n²) geodesic energy / CCD kernels use G_THREADS independently.
  const int thread_count =
      std::clamp(n / 2000, 1, (int)std::thread::hardware_concurrency());
  std::cout << "  Threads  : " << thread_count << " (Repulsor), " << G_THREADS
            << " (energy/CCD)\n";
  auto mesh_ptr = build_mesh(mesh_fac, v4, n, thread_count);
  auto &M = *mesh_ptr;

  // ── Gradient buffers ─────────────────────────────────────────────────
  Mesh_T::CotangentVector_T diff(n, AMB_DIM);  // L² gradient (cotangent)
  Mesh_T::TangentVector_T g_sob(n, AMB_DIM);   // Sobolev gradient
  Mesh_T::TangentVector_T new_pos(n, AMB_DIM); // updated coordinates

  // ── Log file ─────────────────────────────────────────────────────────
  std::ofstream log(log_path);
  if (!log)
    throw std::runtime_error("Cannot write: " + log_path);
  log << "iteration,energy,gradient_norm,step_size\n";
  log << std::setprecision(12) << std::scientific;

  // ── Trajectory file for the live viewer (--frames K) ───────────────────
  std::ofstream traj;
  if (FRAME_EVERY > 0 && !TRAJ_PATH.empty()) {
    traj.open(TRAJ_PATH, std::ios::trunc);
    if (!traj)
      std::cerr << "Warning: cannot write trajectory " << TRAJ_PATH << "\n";
    else
      std::cout << "  Frames   : every " << FRAME_EVERY << " iters → "
                << TRAJ_PATH << "\n";
  }

  Real alpha = alpha0;

  // ── Constants hoisted outside the iteration loop ─────────────────────────
  constexpr Real ARMIJO_C = 1e-4;   // sufficient-decrease constant
  constexpr Real ALPHA_MIN = 1e-15; // step-size floor
  constexpr int MAX_BT = 50;        // max backtracking halvings

  // ── Convergence criteria ─────────────────────────────────────────────────
  // The line search enforces strict descent of the GEODESIC energy (the
  // objective we log and report). We declare convergence as soon as:
  //   (a) no step size down to ALPHA_MIN can decrease it ("won't budge"), or
  //   (b) the relative decrease stays below REL_TOL for MAX_NO_PROGRESS
  //       consecutive iterations, or
  //   (c) the projected gradient norm vanishes.
  constexpr Real REL_TOL = 1e-10;
  constexpr int MAX_NO_PROGRESS = 15;
  int no_progress = 0;

  // ── CCD: dynamic d_safe ─────────────────────────────────────────────────
  // Recomputed every iteration from current edge lengths.
  Real avg_edge_len = 0.0;

  // ── Momentum (Riemannian heavy-ball) ─────────────────────────────────────
  // `mom` holds the tangential displacement applied on the previous accepted
  // step, expressed at the current iterate. Adding μ·mom to the gradient step
  // accelerates the flow through the long, shallow symmetry-saddle valleys
  // where plain descent stalls for thousands of iterations. The line search
  // still guarantees STRICT energy decrease (we fall back to a pure gradient
  // step whenever the momentum term would raise the energy), so the reported
  // energy stays monotone — essential for the minimiser argument.
  constexpr Real MU = 0.9; // momentum coefficient
  std::vector<Real> mom(n * 4, 0.0);

  // ── One-time symmetry-breaking perturbation ──────────────────────────────
  // A freshly generated torus knot sits at a SYMMETRIC saddle of the energy,
  // where ∇E ≈ 0 by symmetry. Plain descent cannot leave it; momentum from
  // rest cannot either. A tiny random tangential kick breaks the symmetry so
  // the flow can roll off the saddle toward the true minimiser. Deterministic
  // seed → reproducible runs.
  {
    std::mt19937 rng(12345u);
    std::normal_distribution<Real> gauss(0.0, 1.0);
    Real edge0 = 0.0;
    for (int i = 0; i < n; ++i) {
      int ip = (i + 1) % n;
      Real e[4];
      for (int d = 0; d < 4; ++d) e[d] = v4[4 * ip + d] - v4[4 * i + d];
      edge0 += norm_4(e);
    }
    edge0 /= n;
    // Keep the kick well below the closest strand approach so it cannot
    // itself cause a self-crossing.
    Real gap0 = global_min_gap(v4, n);
    const Real kick = std::min(1e-3 * edge0, 0.1 * gap0);
    for (int k = 0; k < n; ++k) {
      Real r[4] = {gauss(rng), gauss(rng), gauss(rng), gauss(rng)};
      Real t[4];
      proj_tangent(v4.data() + 4 * k, r, t); // keep perturbation on S³
      Real q[4];
      for (int d = 0; d < 4; ++d) q[d] = v4[4 * k + d] + kick * t[d];
      normalise4(q);
      for (int d = 0; d < 4; ++d) v4[4 * k + d] = q[d];
    }
    for (int i = 0; i < n * AMB_DIM; ++i) new_pos.data()[i] = v4[i];
    M.SemiStaticUpdate(new_pos.data());
  }

  // Geodesic energy at the current iterate. Updated incrementally on step
  // acceptance so it is computed exactly once per configuration.
  Real E = compute_energy_ohara(v4, n);

  // ── Tunnel-proof step cap ────────────────────────────────────────────────
  // No static collision constraint (it froze the flow far short of the
  // minimiser). Instead we bound every per-step vertex displacement to a
  // fraction of the CURRENT global minimum non-adjacent strand gap. If no
  // vertex moves more than STEP_CAP_FRAC·gap, two strands initially `gap`
  // apart stay > (1 − 2·STEP_CAP_FRAC)·gap > 0 apart — the curve provably
  // cannot pass through itself, while the tangent-point energy barrier does
  // the rest. Higher N shrinks edges, so the gap is resolved finely and the
  // cap rarely bites away from genuine near-contact.
  constexpr Real STEP_CAP_FRAC = 0.25;
  Real g_min = 0.0; // global min non-adjacent strand gap (this iterate)

  // Re-parametrise to uniform geodesic arc length every REPARAM_EVERY steps
  // (global; --reparam K, 0 = off).  Uniform arc length undoes tangential
  // point-clustering, but for a connect-sum it STARVES a shrinking summand of
  // vertices (it gets points ∝ its arc length), which is what lets it drop
  // below the resolution needed to hold its crossings and untie.

  for (int iter = 0; iter <= max_iter; ++iter) {

    if (REPARAM_EVERY > 0 && iter > 0 && iter % REPARAM_EVERY == 0) {
      reparametrize_s3(v4, n);
      for (int i = 0; i < n * AMB_DIM; ++i) new_pos.data()[i] = v4[i];
      M.SemiStaticUpdate(new_pos.data());
      std::fill(mom.begin(), mom.end(), 0.0);
      E = compute_energy_ohara(v4, n);
    }

    // avg edge length (for diagnostics) + global strand gap (for the cap)
    avg_edge_len = 0.0;
    for (int i = 0; i < n; ++i) {
      int ip = (i + 1) % n;
      Real ediff[4];
      for (int d = 0; d < 4; ++d) ediff[d] = v4[4 * ip + d] - v4[4 * i + d];
      avg_edge_len += norm_4(ediff);
    }
    avg_edge_len /= n;
    g_min = global_min_gap(v4, n);
    if (iter == 0)
      std::cout << "  step cap : " << STEP_CAP_FRAC * g_min
                << "  (min strand gap: " << g_min
                << ", avg edge: " << avg_edge_len << ")\n";

    // ── 1. Analytic gradient of the O'Hara energy E^{(2)}_{S³} ────────
    //   This is the exact gradient of the functional the line search judges
    //   (compute_energy_ohara), so |∇E| → 0 is a genuine criticality test.
    //
    //   We first issue one Repulsor Differential call purely to warm the
    //   mesh's metric caches (TangentPointMetric0::Solve below reuses the
    //   BVH/near-far operators it populates); its tangent-point gradient is
    //   then discarded and overwritten by the O'Hara gradient.
    if (n < 50)
      (void)E_ap.Differential(M, diff.data());
    else
      (void)E_bh.Differential(M, diff.data());

    compute_gradient_ohara(v4, n, diff.data());

    // Project diff onto T_{x_k}S³
    for (int k = 0; k < n; ++k) {
      Real tmp[4];
      proj_tangent(v4.data() + 4 * k, diff.data() + 4 * k, tmp);
      for (int d = 0; d < 4; ++d)
        diff.data()[4 * k + d] = tmp[d];
    }

    Real gnorm = 0.0;
    for (int i = 0; i < n * AMB_DIM; ++i)
      gnorm += diff.data()[i] * diff.data()[i];
    gnorm = std::sqrt(gnorm);

    log << iter << "," << E << "," << gnorm << "," << alpha << "\n";

    if (traj.is_open() && (iter % FRAME_EVERY == 0 || iter == max_iter))
      write_frame(traj, v4, n, iter, E);

    if (iter % 50 == 0 || iter == max_iter)
      std::cout << "iter " << std::setw(5) << iter
                << "  E_geo=" << std::setprecision(8) << E << "  |g|=" << gnorm
                << "  α=" << alpha << "\n";

    if (gnorm < 1e-10 || iter == max_iter)
      break;

    // ── 3. Sobolev H^{1/2} preconditioning ───────────────────────────
    //   Solve:  metric * g_sob = diff
    //   g_sob is the Sobolev gradient; convergence is resolution-independent.
    // The solve only needs to produce a good descent direction — the line
    // search guards correctness — so a moderate tolerance is sufficient.
    const Int solver_iter = 100;
    const Real solver_tol = 1e-4;

    metric.Solve(M, Real(1), diff.data(), AMB_DIM, Real(0), g_sob.data(),
                 AMB_DIM, AMB_DIM, solver_iter, solver_tol);

    // Project g_sob onto T_{x_k}S³
    for (int k = 0; k < n; ++k) {
      Real tmp[4];
      proj_tangent(v4.data() + 4 * k, g_sob.data() + 4 * k, tmp);
      for (int d = 0; d < 4; ++d)
        g_sob.data()[4 * k + d] = tmp[d];
    }

    // ── 4. Backtracking line search on the O'Hara energy ──────────────
    //
    // Sufficient decrease is enforced on E^{(2)}_{S³} directly:
    //   E(trial) <= E - c · α · <∇E, g_sob>.
    // diff = ∇E (the exact O'Hara gradient), g_sob = A⁻¹∇E.  For a positive-
    // definite metric <∇E, g_sob> > 0 is guaranteed — but the preconditioned
    // GMRES solve can return a non-descent direction when the metric is badly
    // ill-conditioned (e.g. a tight connect-sum neck).  That is a SOLVER
    // failure, not convergence: fall back to the raw projected gradient, which
    // is always a descent direction (<∇E, ∇E> = |∇E|² > 0), so the flow keeps
    // going instead of stalling at a pinched configuration.
    Real descent = 0.0;
    for (int i = 0; i < n * AMB_DIM; ++i)
      descent += diff.data()[i] * g_sob.data()[i];

    if (descent <= 0.0) {
      for (int i = 0; i < n * AMB_DIM; ++i)
        g_sob.data()[i] = diff.data()[i];   // fall back to steepest descent
      descent = gnorm * gnorm;              // = <∇E, ∇E> > 0
      if (iter % 50 == 0)
        std::cout << "  [precond fallback: H^{1/2} solve non-descent; "
                     "using raw gradient]\n";
    }

    // ── 4a. Compute maximum displacement (after any fallback) ─────────
    Real max_g_sob = 0.0;
    for (int k = 0; k < n; ++k) {
      Real g_norm = norm_4(g_sob.data() + 4 * k);
      if (g_norm > max_g_sob) max_g_sob = g_norm;
    }

    std::vector<Real> v4_cand(n * 4);

    // ── 4a. Transport previous momentum into the current tangent space ─
    // mom was a tangent vector at the previous iterate; project it onto
    // T_{x}S³ here and rescale to preserve its length (parallel-transport
    // approximation, exact to first order for the small steps we take).
    Real mom_norm2 = 0.0;
    for (int i = 0; i < n * AMB_DIM; ++i) mom_norm2 += mom[i] * mom[i];
    if (mom_norm2 > EPS) {
      Real new_norm2 = 0.0;
      for (int k = 0; k < n; ++k) {
        Real t[4];
        proj_tangent(v4.data() + 4 * k, mom.data() + 4 * k, t);
        for (int d = 0; d < 4; ++d) { mom[4 * k + d] = t[d]; new_norm2 += t[d] * t[d]; }
      }
      if (new_norm2 > EPS) {
        Real s = std::sqrt(mom_norm2 / new_norm2);
        for (int i = 0; i < n * AMB_DIM; ++i) mom[i] *= s;
      } else {
        std::fill(mom.begin(), mom.end(), 0.0);
      }
    }

    // ── 4b. Tunnel-proof cap ──────────────────────────────────────────
    // Budget = STEP_CAP_FRAC · g_min for the worst-moving vertex. Split it
    // half/half between momentum and gradient when momentum is active; the
    // pure-gradient fallback gets the full budget. Cap the (already
    // transported) momentum here so μ·max|mom| ≤ ½·max_move.
    const Real max_move = STEP_CAP_FRAC * g_min;
    {
      Real max_mom = 0.0;
      for (int k = 0; k < n; ++k)
        max_mom = std::max(max_mom, norm_4(mom.data() + 4 * k));
      if (MU * max_mom > 0.5 * max_move && max_mom > EPS) {
        Real s = (0.5 * max_move) / (MU * max_mom);
        for (int i = 0; i < n * AMB_DIM; ++i) mom[i] *= s;
        mom_norm2 *= s * s;
      }
    }

    bool accepted = false;
    Real E_new = E;

    // Line search tries the momentum-augmented step first; if it cannot
    // produce strict decrease (momentum overshot), it retries with pure
    // gradient (use_momentum=false), which is guaranteed to descend.
    for (int pass = 0; pass < 2 && !accepted; ++pass) {
      const bool use_momentum = (pass == 0) && (mom_norm2 > EPS);

      Real alpha_try = alpha;
      // Cap the gradient part so the total per-vertex move ≤ max_move,
      // which makes self-passage geometrically impossible (see §4 cap note).
      if (max_g_sob > EPS) {
        Real budget = use_momentum ? 0.5 * max_move : max_move;
        Real alpha_cap = budget / max_g_sob;
        if (alpha_try > alpha_cap) alpha_try = alpha_cap;
      }

      for (int bt = 0; bt < MAX_BT && alpha_try >= ALPHA_MIN; ++bt) {

        // trial displacement: -α·g_sob (+ μ·mom if this pass uses momentum)
        for (int k = 0; k < n; ++k) {
          Real pk[4], step[4], q[4];
          for (int d = 0; d < 4; ++d) pk[d] = v4[4 * k + d];
          for (int d = 0; d < 4; ++d) {
            step[d] = -alpha_try * g_sob.data()[4 * k + d];
            if (use_momentum) step[d] += MU * mom[4 * k + d];
          }
          exp_map(pk, step, q);
          normalise4(q);
          for (int d = 0; d < 4; ++d) v4_cand[4 * k + d] = q[d];
        }

        // strict sufficient decrease of the geodesic energy
        Real E_geo_trial = compute_energy_ohara(v4_cand, n);
        if (E_geo_trial > E - ARMIJO_C * alpha_try * descent) {
          alpha_try *= 0.5;
          continue;
        }

        // ── Tunnel guard: ℝ³ continuous collision detection (rigorous) ──
        // Reject the step iff some non-adjacent edge pair would pass THROUGH
        // another as the curve moves from v4 to v4_cand (tested in the ℝ³
        // stereographic image, where knotting actually lives — the old ℝ⁴-chord
        // gap test let crossings slip through). Forbidding the crossing event
        // itself is tunnel-proof at any step size and never freezes.
        if (any_self_passage(v4, v4_cand, n)) {
          alpha_try *= 0.5;
          continue;
        }

        // accept: record the displacement as the new momentum, push coords
        for (int k = 0; k < n; ++k) {
          for (int d = 0; d < 4; ++d) {
            Real s = -alpha_try * g_sob.data()[4 * k + d];
            if (use_momentum) s += MU * mom[4 * k + d];
            mom[4 * k + d] = s;
          }
        }
        v4 = v4_cand;
        E_new = E_geo_trial;
        for (int i = 0; i < n * AMB_DIM; ++i) new_pos.data()[i] = v4[i];
        M.SemiStaticUpdate(new_pos.data());
        accepted = true;
        alpha = std::min(alpha_try * 1.2, alpha0 * 10.0);
        break;
      }
    }

    if (!accepted) {
      // No step size down to ALPHA_MIN decreases the geodesic energy along
      // the (pure-gradient) descent direction: the configuration won't budge.
      // This IS the stopping point — a genuine critical configuration.
      std::cout << "\n  *** CONVERGED at iter " << iter
                << " (line search exhausted: no step decreases E_geo) ***\n";
      std::cout << "  E_geo = " << std::setprecision(10) << E
                << "   |g| = " << gnorm << "\n";
      break;
    }

    // ── 5. Progress check ──────────────────────────────────────────────
    no_progress = ((E - E_new) <= REL_TOL * std::fabs(E)) ? no_progress + 1 : 0;
    E = E_new;
    if (no_progress >= MAX_NO_PROGRESS) {
      std::cout << "\n  *** CONVERGED at iter " << iter
                << " (relative decrease < " << REL_TOL << " for "
                << MAX_NO_PROGRESS << " consecutive iterations) ***\n";
      break;
    }
  }

  log.close();
}

// ═══════════════════════════════════════════════════════════════════════════
// §7  main
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char **argv) {
  Profiler::Clear();

  G_THREADS = std::max(1u, std::thread::hardware_concurrency());

  if (argc < 4) {
    std::cerr << "Usage: energy_s3 <input.vect> <energy_log.csv> <output.vect>"
                 " [max_iter=500] [step=0.01] [--ccd]\n";
    return 1;
  }

  bool gradcheck = false;
  // Parse flags anywhere in args: --ccd, --gradcheck, --frames K
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ccd" || a == "--gradcheck" || a == "--no-normalize") {
      if (a == "--ccd") CCD_ENABLED = true;
      else if (a == "--no-normalize") NORMALIZE = false;
      else gradcheck = true;
      for (int j = i; j < argc - 1; ++j) argv[j] = argv[j + 1];
      --argc;
      --i;
    } else if (a == "--frames") {
      // Consume the flag AND its integer argument.
      FRAME_EVERY = (i + 1 < argc) ? std::max(1, std::atoi(argv[i + 1])) : 10;
      int consume = (i + 1 < argc) ? 2 : 1;
      for (int j = i; j < argc - consume; ++j) argv[j] = argv[j + consume];
      argc -= consume;
      --i;
    } else if (a == "--reparam") {
      // Consume the flag AND its integer argument (0 = disable reparametrisation).
      REPARAM_EVERY = (i + 1 < argc) ? std::max(0, std::atoi(argv[i + 1])) : 50;
      int consume = (i + 1 < argc) ? 2 : 1;
      for (int j = i; j < argc - consume; ++j) argv[j] = argv[j + consume];
      argc -= consume;
      --i;
    }
  }

  // ── Gradient self-test: central finite differences vs analytic ──────────
  if (gradcheck) {
    int n = 0;
    std::vector<Real> pts_r3 = read_vect(argv[1], n);
    std::vector<Real> v4(n * 4);
    for (int i = 0; i < n; ++i)
      r3_to_s3(pts_r3.data() + 3 * i, v4.data() + 4 * i);
    std::vector<Real> g(n * 4);
    compute_gradient_ohara(v4, n, g.data());
    // Project analytic gradient onto T_xS³ for fair comparison
    for (int k = 0; k < n; ++k) {
      Real tmp[4];
      proj_tangent(v4.data() + 4 * k, g.data() + 4 * k, tmp);
      for (int d = 0; d < 4; ++d) g[4 * k + d] = tmp[d];
    }
    const Real h = 1e-6;
    Real num = 0.0, den = 0.0, max_abs = 0.0;
    // Sample 40 vertices to keep it fast
    int step = std::max(1, n / 40);
    for (int k = 0; k < n; k += step) {
      for (int d = 0; d < 4; ++d) {
        std::vector<Real> vp = v4, vm = v4;
        vp[4 * k + d] += h; normalise4(vp.data() + 4 * k);
        vm[4 * k + d] -= h; normalise4(vm.data() + 4 * k);
        Real fd = (compute_energy_ohara(vp, n) -
                   compute_energy_ohara(vm, n)) / (2 * h);
        // Tangential part of the finite-difference direction
        Real an = g[4 * k + d];
        num += (fd - an) * (fd - an);
        den += fd * fd;
        max_abs = std::max(max_abs, std::fabs(fd - an));
      }
    }
    std::cout << "Gradient check (n=" << n << ", h=" << h << "):\n"
              << "  relative L2 error : " << std::sqrt(num / (den + EPS)) << "\n"
              << "  max abs error     : " << max_abs << "\n";
    return 0;
  }

  const std::string in_path = argv[1];
  const std::string log_path = argv[2];
  const std::string out_path = argv[3];
  const int max_iter = (argc >= 5) ? std::stoi(argv[4]) : 500;
  const Real alpha0 = (argc >= 6) ? std::stod(argv[5]) : 0.01;

  // Trajectory for the live viewer goes next to the output .vect.
  if (FRAME_EVERY > 0) {
    auto slash = out_path.find_last_of("/\\");
    TRAJ_PATH = (slash == std::string::npos ? "" : out_path.substr(0, slash + 1)) +
                "trajectory.jsonl";
  }

  std::cout << "S³ O'Hara Energy Minimiser  E^(2)_{S³}\n";
  std::cout << "═══════════════════════════════════════════\n";
  std::cout << "  Input    : " << in_path << "\n";
  std::cout << "  Log      : " << log_path << "\n";
  std::cout << "  Output   : " << out_path << "\n";
  std::cout << "  Max iter : " << max_iter << "\n";
  std::cout << "  α₀       : " << alpha0 << "\n";
  std::cout << "  BH θ     : " << BH_THETA << "  (n≥50)\n";
  std::cout << "  Safety   : tunnel-proof step cap (0.25·min strand gap)"
            << (CCD_ENABLED ? "  [--ccd: no-op, superseded]" : "") << "\n\n";

  // Load .vect (R³)
  int n = 0;
  std::vector<Real> pts_r3 = read_vect(in_path, n);
  std::cout << "Loaded " << n << " vertices.\n";
  std::cout << (n >= 50 ? "Using Barnes-Hut (TangentPointEnergy0).\n"
                        : "Using AllPairs   (TangentPointEnergy_AllPairs).\n");

  // ── Normalise placement on S³ ────────────────────────────────────────────
  // The inverse stereographic lift is NOT scale-invariant: large-radius R³
  // points map near the north pole, squashing the whole knot into a tiny
  // polar cap where every strand gap becomes microscopic — which throttles the
  // tunnel-proof step cap to a crawl (the ×20-scaled composites had gap < edge
  // length).  Centre the curve and scale it to RMS radius 1 so it lifts to a
  // well-spread band around the equator (empirically maximises the strand gap;
  // for an N=1000 Granny this enlarges the gap ~11×).  Knot type and the energy
  // minimiser are unchanged — only the starting placement and convergence speed.
  //
  // BUT this is an ℝ³ similarity, NOT an S³ isometry: it distorts a knot that was
  // already generated on S³.  A torus knot T(p,q) is built on the Clifford torus
  // (uniform, symmetric, x₄∈[−0.707,0.707]) and projected to ℝ³; the lift below
  // inverts the projection EXACTLY, so without this step it lands back on the
  // pristine Clifford torus.  With it, the curve comes back distorted (edge
  // spread 1.0→1.5, strand gap shrunk ~20%, pushed off-centre) — a worse start.
  // So: skip it for torus knots (`--no-normalize`); keep it for the ℝ³-scale
  // composites that genuinely need re-placing near the equator.
  if (NORMALIZE) {
    Real c[3] = {0, 0, 0};
    for (int i = 0; i < n; ++i)
      for (int d = 0; d < 3; ++d) c[d] += pts_r3[3 * i + d];
    for (int d = 0; d < 3; ++d) c[d] /= n;
    Real ms = 0.0;
    for (int i = 0; i < n; ++i)
      for (int d = 0; d < 3; ++d) {
        Real y = pts_r3[3 * i + d] - c[d];
        ms += y * y;
      }
    Real rms = std::sqrt(ms / n);
    Real s = (rms > EPS) ? 1.0 / rms : 1.0;
    for (int i = 0; i < n; ++i)
      for (int d = 0; d < 3; ++d)
        pts_r3[3 * i + d] = (pts_r3[3 * i + d] - c[d]) * s;
    std::cout << "Normalised R³ placement: centroid removed, RMS radius "
              << std::setprecision(4) << rms << " → 1.0\n";
  } else {
    std::cout << "R³ normalisation SKIPPED (--no-normalize): "
                 "lifting input straight to S³ (pristine Clifford torus).\n";
  }

  // Lift to S³ via inverse stereographic projection
  std::vector<Real> v4(n * 4);
  for (int i = 0; i < n; ++i)
    r3_to_s3(pts_r3.data() + 3 * i, v4.data() + 4 * i);

  Real E0 = compute_energy_ohara(v4, n);
  std::cout << "Initial E^(2)_{S³} energy: " << std::setprecision(10) << E0
            << "\n\n";

  // Run
  gradient_descent(v4, n, log_path, max_iter, alpha0);

  Real Ef = compute_energy_ohara(v4, n);
  std::cout << "\nFinal   E^(2)_{S³} energy : " << std::setprecision(10) << Ef
            << "\n";
  std::cout << "Reduction             : " << std::setprecision(4) << E0 / Ef
            << "×  (" << 100.0 * (1.0 - Ef / E0) << "% decrease)\n";

  // Project back to R³ and write
  for (int i = 0; i < n; ++i)
    s3_to_r3(v4.data() + 4 * i, pts_r3.data() + 3 * i);
  write_vect(out_path, pts_r3, n);
  std::cout << "Written: " << out_path << "\n";
  return 0;
}
