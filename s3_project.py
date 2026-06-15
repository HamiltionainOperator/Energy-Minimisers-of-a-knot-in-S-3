#!/usr/bin/env python3
"""
Stereographic projection utilities between S³ ⊂ ℝ⁴ and ℝ³.

Conventions
-----------
  S³ = { x ∈ ℝ⁴ : ‖x‖ = 1 }
  North pole  N = (0, 0, 0, 1)

  S3toR3(x) : S³ → ℝ³         (stereographic projection from N)
      x = (x₁,x₂,x₃,x₄)  ↦  (x₁, x₂, x₃) / (1 − x₄)
      Undefined at N itself.

  R3toS3(y) : ℝ³ → S³         (inverse stereographic projection)
      y = (y₁,y₂,y₃)  ↦  (2y, |y|²−1) / (|y|²+1)

Both functions accept batched input with an arbitrary leading shape (..., d).

Additional utilities
--------------------
  geodesic_distance_s3   — arc-length d(x,y) = arccos(<x,y>)
  log_map_s3             — logarithm map log_x(y) (tangent vector at x)
  exp_map_s3             — exponential map exp_x(v) (point on S³)
  parallel_transport_s3  — parallel transport of v from x to y
  proj_tangent_s3        — orthogonal projection onto T_x S³
  normalize_s3           — renormalise columns of (...,4) arrays onto S³

Run as script for a quick self-test:
    python3 s3_project.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core projections
# ---------------------------------------------------------------------------

def S3toR3(pts: np.ndarray) -> np.ndarray:
    """
    Stereographic projection S³ → ℝ³ from north pole N = (0,0,0,1).

    Parameters
    ----------
    pts : (..., 4) array of unit vectors on S³

    Returns
    -------
    (..., 3) array in ℝ³

    Raises
    ------
    ValueError if any point is at or near N.
    """
    pts = np.asarray(pts, dtype=float)
    x4 = pts[..., 3]
    denom = 1.0 - x4
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError("Point(s) at/near north pole N; projection undefined.")
    return pts[..., :3] / denom[..., np.newaxis]


def R3toS3(pts: np.ndarray) -> np.ndarray:
    """
    Inverse stereographic projection ℝ³ → S³.

    Parameters
    ----------
    pts : (..., 3) array in ℝ³

    Returns
    -------
    (..., 4) array of unit points on S³
    """
    pts   = np.asarray(pts, dtype=float)
    r2    = np.sum(pts ** 2, axis=-1, keepdims=True)   # ‖y‖²
    denom = r2 + 1.0
    xyz   = 2.0 * pts / denom
    w     = (r2 - 1.0) / denom
    return np.concatenate([xyz, w], axis=-1)


# ---------------------------------------------------------------------------
# Geodesic distance
# ---------------------------------------------------------------------------

def geodesic_distance_s3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Geodesic (great-circle) distance on S³.

    d(a, b) = arccos( clamp(<a,b>, −1, 1) )  ∈ [0, π]

    Parameters
    ----------
    a, b : (..., 4) arrays of unit points on S³

    Returns
    -------
    (...,) array of angles in [0, π]
    """
    a   = np.asarray(a, dtype=float)
    b   = np.asarray(b, dtype=float)
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.arccos(dot)


# ---------------------------------------------------------------------------
# Logarithm and exponential maps
# ---------------------------------------------------------------------------

def log_map_s3(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Riemannian logarithm map:  log_x(y) ∈ T_x S³.

    Returns the unique tangent vector v at x such that exp_x(v) = y and
    ‖v‖ = d(x,y).

    Parameters
    ----------
    x : (..., 4) base points on S³
    y : (..., 4) target points on S³
    eps : threshold below which x ≈ y is detected

    Returns
    -------
    (..., 4) tangent vectors at x (orthogonal to x)
    """
    x    = np.asarray(x, dtype=float)
    y    = np.asarray(y, dtype=float)
    dot  = np.clip(np.sum(x * y, axis=-1, keepdims=True), -1.0, 1.0)
    # Tangent component of y at x
    tang = y - dot * x
    norm = np.linalg.norm(tang, axis=-1, keepdims=True)
    theta = np.arccos(np.squeeze(dot, axis=-1))   # geodesic distance

    # Safe normalise: where norm ~ 0 the log is the zero vector
    safe_norm = np.where(norm < eps, 1.0, norm)
    direction = tang / safe_norm
    scale     = np.where(norm[..., 0] < eps, 0.0, theta)
    return direction * scale[..., np.newaxis]


def exp_map_s3(x: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Riemannian exponential map:  exp_x(v) ∈ S³.

    v must be tangent at x (orthogonal to x), but a small component along x
    is silently projected out for robustness.

    Parameters
    ----------
    x : (..., 4) base points on S³
    v : (..., 4) tangent vectors at x

    Returns
    -------
    (..., 4) points on S³
    """
    x    = np.asarray(x, dtype=float)
    v    = np.asarray(v, dtype=float)
    # Project out any radial component
    v    = v - np.sum(v * x, axis=-1, keepdims=True) * x
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    safe = np.where(norm < eps, 1.0, norm)
    unit = v / safe
    cos_ = np.cos(norm)
    sin_ = np.sin(norm)
    return cos_ * x + sin_ * unit


# ---------------------------------------------------------------------------
# Parallel transport
# ---------------------------------------------------------------------------

def parallel_transport_s3(x: np.ndarray, v: np.ndarray,
                           y: np.ndarray) -> np.ndarray:
    """
    Parallel transport of tangent vector v (at x) along the geodesic to y.

    Uses the closed-form formula for parallel transport on S^n:

        PT(v) = v − <v, e>( sin θ · x + (1 − cos θ) · e )

    where e = log_x(y)/θ is the unit geodesic direction and θ = d(x,y).

    Parameters
    ----------
    x : (..., 4) base points on S³
    v : (..., 4) tangent vectors at x (orthogonal to x)
    y : (..., 4) target points on S³

    Returns
    -------
    (..., 4) tangent vectors at y (orthogonal to y)
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    y = np.asarray(y, dtype=float)

    log_xy = log_map_s3(x, y)                          # (..., 4)
    theta  = np.linalg.norm(log_xy, axis=-1, keepdims=True)  # (..., 1)

    # Unit direction of geodesic
    safe   = np.where(theta < 1e-12, 1.0, theta)
    e      = log_xy / safe

    v_par  = np.sum(v * e, axis=-1, keepdims=True)     # component along e
    transported = (v
                   - v_par * np.sin(theta) * x
                   - v_par * (1.0 - np.cos(theta)) * e)
    return transported


# ---------------------------------------------------------------------------
# Tangent-space projection and normalisation
# ---------------------------------------------------------------------------

def proj_tangent_s3(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Project v onto T_x S³ = { w : <x,w> = 0 }.

    Parameters
    ----------
    x : (..., 4) points on S³
    v : (..., 4) vectors to project

    Returns
    -------
    (..., 4) tangent vectors
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    return v - np.sum(v * x, axis=-1, keepdims=True) * x


def normalize_s3(pts: np.ndarray) -> np.ndarray:
    """Project arbitrary ℝ⁴ points onto S³ by L² normalisation."""
    pts   = np.asarray(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=-1, keepdims=True)
    return pts / norms


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy.testing as npt

    print("=== s3_project.py self-test ===\n")

    # Round-trip test
    y = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.5, 0.5, 0.5],
                  [10.0, -3.0, 0.0]])
    x = R3toS3(y)
    npt.assert_allclose(np.linalg.norm(x, axis=-1), 1.0, atol=1e-12)
    y_back = S3toR3(x)
    npt.assert_allclose(y_back, y, atol=1e-12)
    print("R3 → S3 → R3 round-trip: OK")

    # Geodesic distance
    a = R3toS3(np.array([[1.0, 0.0, 0.0]]))[0]
    b = R3toS3(np.array([[0.0, 1.0, 0.0]]))[0]
    d = geodesic_distance_s3(a, b)
    print(f"Geodesic distance between two test points: {d:.6f} rad")

    # exp ∘ log = identity
    log_v = log_map_s3(a, b)
    b_rec = exp_map_s3(a, log_v)
    npt.assert_allclose(b_rec, b, atol=1e-12)
    print("exp(log_x(y)) = y: OK")

    # Parallel transport preserves norm and lands in T_y S³
    v = proj_tangent_s3(a, np.array([0.0, 0.0, 1.0, 0.0]))
    vt = parallel_transport_s3(a, v, b)
    npt.assert_allclose(np.dot(vt, b), 0.0, atol=1e-11)
    npt.assert_allclose(np.linalg.norm(vt), np.linalg.norm(v), atol=1e-11)
    print("Parallel transport: tangency and norm preserved: OK")

    print("\nAll tests passed.")
