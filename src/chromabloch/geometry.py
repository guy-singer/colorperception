"""Hilbert/Klein disk geometry and gyrovector operations.

The Bloch disk D with the Klein (Beltrami-Klein) model of hyperbolic geometry.
The Hilbert distance coincides with the Klein hyperbolic distance.

References:
    Definition: Hilbert Distance
    Proposition: Klein Distance Formula
    Definition: Klein Gyroaddition
    Theorem: Distance via Gyroaddition
"""

from __future__ import annotations

import numpy as np

# Numerical safety margin for boundary
_BOUNDARY_MARGIN = 1e-15


def _euclidean_norm(v: np.ndarray) -> np.ndarray:
    """Compute Euclidean norm ||v|| along last axis (NOT squared)."""
    return np.linalg.norm(v, axis=-1)


def _dot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute dot product along last axis."""
    return np.sum(u * v, axis=-1)


def hilbert_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute Hilbert distance d_H(p, q) on the Klein disk.

    Proposition: Klein Distance Formula.

    d_H(p, q) = arcosh((1 - ⟨p,q⟩) / sqrt((1 - ||p||²)(1 - ||q||²)))

    Parameters
    ----------
    p : array-like
        First point(s) in D, shape (..., 2).
    q : array-like
        Second point(s) in D, shape (..., 2).

    Returns
    -------
    d : ndarray
        Hilbert distances, shape (...).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    rp2 = np.sum(p * p, axis=-1)
    rq2 = np.sum(q * q, axis=-1)
    pq = _dot(p, q)

    # Clamp norms away from 1 for numerical stability
    rp2 = np.clip(rp2, 0.0, 1.0 - _BOUNDARY_MARGIN)
    rq2 = np.clip(rq2, 0.0, 1.0 - _BOUNDARY_MARGIN)

    # Argument to arcosh
    numerator = 1.0 - pq
    denominator = np.sqrt((1.0 - rp2) * (1.0 - rq2))

    # arcosh argument must be >= 1
    arg = numerator / denominator
    arg = np.maximum(arg, 1.0)

    return np.arccosh(arg)


def hilbert_distance_from_origin(v: np.ndarray) -> np.ndarray:
    """Compute Hilbert distance d_H(0, v) = arctanh(||v||).

    Special case verification from the Klein formula.

    Parameters
    ----------
    v : array-like
        Point(s) in D, shape (..., 2).

    Returns
    -------
    d : ndarray
        Distance from origin, shape (...).
    """
    v = np.asarray(v, dtype=float)
    r = _euclidean_norm(v)

    # Clamp for numerical stability
    r = np.clip(r, 0.0, 1.0 - _BOUNDARY_MARGIN)

    return np.arctanh(r)


def gamma_factor(u: np.ndarray) -> np.ndarray:
    """Compute Lorentz factor Γ_u = (1 - ||u||²)^(-1/2).

    Definition: Klein Gyroaddition (Lorentz factor).

    Note: This Γ is unrelated to the opponent parameter γ.

    Parameters
    ----------
    u : array-like
        Point(s) in D, shape (..., 2).

    Returns
    -------
    Gamma : ndarray
        Lorentz factors, shape (...).
    """
    u = np.asarray(u, dtype=float)
    r2 = np.sum(u * u, axis=-1)

    # Clamp for numerical stability
    r2 = np.clip(r2, 0.0, 1.0 - _BOUNDARY_MARGIN)

    return 1.0 / np.sqrt(1.0 - r2)


def klein_gyroadd(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute Klein gyroaddition u ⊕ v.

    Definition: Klein Gyroaddition.

    u ⊕ v = (1/(1 + ⟨u,v⟩)) * [u + v/Γ_u + (Γ_u/(1+Γ_u)) * ⟨u,v⟩ * u]

    where Γ_u = (1 - ||u||²)^(-1/2).

    Parameters
    ----------
    u : array-like
        First operand in D, shape (..., 2).
    v : array-like
        Second operand in D, shape (..., 2).

    Returns
    -------
    result : ndarray
        u ⊕ v, shape (..., 2), with ||result|| < 1.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Compute ⟨u, v⟩
    uv = _dot(u, v)

    # Compute Γ_u
    u_euclidean_norm = np.sum(u * u, axis=-1)
    u_euclidean_norm = np.clip(u_euclidean_norm, 0.0, 1.0 - _BOUNDARY_MARGIN)
    Gamma_u = 1.0 / np.sqrt(1.0 - u_euclidean_norm)

    # Denominator: 1 + ⟨u, v⟩
    denom = 1.0 + uv

    # Handle potential division issues (shouldn't happen for valid inputs)
    denom = np.maximum(denom, _BOUNDARY_MARGIN)

    # Compute the three terms
    # Term 1: u
    term1 = u

    # Term 2: v / Γ_u
    term2 = v / Gamma_u[..., None]

    # Term 3: (Γ_u / (1 + Γ_u)) * ⟨u,v⟩ * u
    coeff = (Gamma_u / (1.0 + Gamma_u)) * uv
    term3 = coeff[..., None] * u

    # Combine
    result = (term1 + term2 + term3) / denom[..., None]

    # Ensure result stays in open disk
    result_norm = _euclidean_norm(result)
    mask = result_norm >= 1.0
    if np.any(mask):
        scale = (1.0 - _BOUNDARY_MARGIN) / result_norm[mask]
        result[mask] *= scale[..., None]

    return result


def boundary_points_on_line(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute boundary intersection points for the line through p and q.

    Proposition: Boundary Point Computation.

    Solve ||p + t(q-p)||² = 1 for t.
    Quadratic: At² + Bt + C = 0 where
        A = ||q-p||²
        B = 2⟨p, q-p⟩
        C = ||p||² - 1

    Parameters
    ----------
    p : array-like
        First point in D, shape (..., 2).
    q : array-like
        Second point in D, shape (..., 2).

    Returns
    -------
    a_minus, a_plus : tuple of ndarray
        Boundary points, shape (..., 2).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    d = q - p  # direction

    A = np.sum(d * d, axis=-1)
    B = 2.0 * _dot(p, d)
    C = np.sum(p * p, axis=-1) - 1.0

    discriminant = B * B - 4.0 * A * C
    discriminant = np.maximum(discriminant, 0.0)  # numerical safety
    sqrt_disc = np.sqrt(discriminant)

    # Handle A = 0 (p = q) case
    with np.errstate(divide="ignore", invalid="ignore"):
        t_minus = (-B - sqrt_disc) / (2.0 * A)
        t_plus = (-B + sqrt_disc) / (2.0 * A)

    # Expand dimensions for broadcasting
    t_minus = t_minus[..., None]
    t_plus = t_plus[..., None]

    a_minus = p + t_minus * d
    a_plus = p + t_plus * d

    return a_minus, a_plus


def hilbert_distance_crossratio(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute Hilbert distance via cross-ratio definition.
    
    Definition: Hilbert Distance (Cross-Ratio Form).
    
    d_H(p, q) = (1/2) * |log((|a⁺-p| * |a⁻-q|) / (|a⁺-q| * |a⁻-p|))|
    
    where a⁺, a⁻ are the boundary intersection points of the line through p, q.
    
    This provides an INDEPENDENT validation of the Klein formula,
    as it computes the distance via a completely different method.
    
    Parameters
    ----------
    p : array-like
        First point in D, shape (..., 2).
    q : array-like
        Second point in D, shape (..., 2).
        
    Returns
    -------
    d : ndarray
        Hilbert distances, shape (...).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Get boundary intersection points
    a_minus, a_plus = boundary_points_on_line(p, q)
    
    # Compute Euclidean distances to boundary points
    # |a⁺ - p|, |a⁺ - q|, |a⁻ - p|, |a⁻ - q|
    d_plus_p = _euclidean_norm(a_plus - p)
    d_plus_q = _euclidean_norm(a_plus - q)
    d_minus_p = _euclidean_norm(a_minus - p)
    d_minus_q = _euclidean_norm(a_minus - q)
    
    # Cross-ratio: (|a⁺-p| * |a⁻-q|) / (|a⁺-q| * |a⁻-p|)
    # Handle numerical edge cases
    numerator = d_plus_p * d_minus_q
    denominator = d_plus_q * d_minus_p
    
    # Avoid division by zero
    denominator = np.maximum(denominator, _BOUNDARY_MARGIN)
    
    cross_ratio = numerator / denominator
    cross_ratio = np.maximum(cross_ratio, _BOUNDARY_MARGIN)
    
    # Hilbert distance = (1/2) * |log(cross_ratio)|
    return 0.5 * np.abs(np.log(cross_ratio))


__all__ = [
    "hilbert_distance",
    "hilbert_distance_crossratio",
    "hilbert_distance_from_origin",
    "gamma_factor",
    "klein_gyroadd",
    "boundary_points_on_line",
]
