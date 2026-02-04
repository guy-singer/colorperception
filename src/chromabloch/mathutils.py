"""Mathematical utilities for the attainable chromaticity region.

The attainable region in chromaticity space (for ε=0) is:
    {(u1, u2) : -γ/w_M < u1 < 1/w_L, u2 > g(u1)}

where g(u1) = -β/Δ * (γ + 1 + (w_M - w_L)*u1) is the affine boundary.

References:
    Proposition: Exact Attainable Region for ε = 0
    Corollary: Attainable Region for ε > 0
"""

from __future__ import annotations

import numpy as np

from .params import Theta


def g_boundary(u1: np.ndarray, theta: Theta) -> np.ndarray:
    """Compute the lower boundary function g(u1).

    Proposition: Exact Attainable Region for ε = 0.

    g(u1) = -β/Δ * (γ + 1 + (w_M - w_L)*u1)

    This is the affine function defining the lower boundary of the
    attainable chromaticity region.

    Parameters
    ----------
    u1 : array-like
        First chromaticity coordinate values.
    theta : Theta
        Parameter set.

    Returns
    -------
    g : ndarray
        Boundary values g(u1).
    """
    u1 = np.asarray(u1, dtype=float)
    Delta = theta.Delta

    return -(theta.beta / Delta) * (
        theta.gamma + 1.0 + (theta.w_M - theta.w_L) * u1
    )


def u1_bounds(theta: Theta) -> tuple[float, float]:
    """Return the u1 bounds: (-γ/w_M, 1/w_L).

    Proposition: Bounds for u1^(0).

    Parameters
    ----------
    theta : Theta
        Parameter set.

    Returns
    -------
    lower, upper : tuple of float
        The bounds for u1.
    """
    lower = -theta.gamma / theta.w_M
    upper = 1.0 / theta.w_L
    return lower, upper


def in_attainable_region_u(u: np.ndarray, theta: Theta, tol: float = 0.0) -> np.ndarray:
    """Check if chromaticity points are in the attainable region.

    Proposition: Exact Attainable Region for ε = 0.

    A point u = (u1, u2) is attainable iff:
        -γ/w_M < u1 < 1/w_L  AND  u2 > g(u1)

    Parameters
    ----------
    u : array-like
        Chromaticity coordinates, shape (..., 2).
    theta : Theta
        Parameter set.
    tol : float
        Tolerance for boundary checks (use positive value to allow
        numerical margin).

    Returns
    -------
    in_region : ndarray of bool
        Shape (...).
    """
    u = np.asarray(u, dtype=float)
    u1 = u[..., 0]
    u2 = u[..., 1]

    lower, upper = u1_bounds(theta)
    g_val = g_boundary(u1, theta)

    in_u1_range = (u1 > lower - tol) & (u1 < upper + tol)
    above_boundary = u2 > g_val - tol

    return in_u1_range & above_boundary


def sample_attainable_region(
    theta: Theta,
    n_samples: int,
    rng: np.random.Generator | None = None,
    margin: float = 0.01,
) -> np.ndarray:
    """Generate random samples from the attainable chromaticity region.

    Samples u1 uniformly in the valid interval and u2 above g(u1).

    Parameters
    ----------
    theta : Theta
        Parameter set.
    n_samples : int
        Number of samples to generate.
    rng : Generator, optional
        Random number generator.
    margin : float
        Margin from boundaries.

    Returns
    -------
    u : ndarray
        Chromaticity samples, shape (n_samples, 2).
    """
    if rng is None:
        rng = np.random.default_rng()

    lower, upper = u1_bounds(theta)

    # Sample u1 with margin
    u1 = rng.uniform(lower + margin, upper - margin, size=n_samples)

    # Compute boundary and sample u2 above it
    g_val = g_boundary(u1, theta)

    # Sample u2 in (g_val + margin, g_val + margin + range)
    # Use exponential offset to get unbounded-above distribution
    u2_offset = rng.exponential(scale=1.0, size=n_samples)
    u2 = g_val + margin + u2_offset

    return np.stack([u1, u2], axis=-1)


def reconstruct_from_attainable(
    u1: np.ndarray,
    u2: np.ndarray,
    theta: Theta,
    M: float = 1.0,
) -> np.ndarray:
    """Reconstruct LMS from attainable chromaticity (sufficiency proof).

    Proposition: Exact Attainable Region for ε = 0 (Sufficiency).

    Given (u1, u2) with u1 in bounds and u2 > g(u1), define:
        t = (γ + w_M*u1) / (1 - w_L*u1)
        D_t = w_L*t + w_M 
        s = β*(t+1) + u2*D_t

    Then for any M > 0: L = t*M, S = s*M.

    Parameters
    ----------
    u1 : array-like
        First chromaticity coordinate.
    u2 : array-like
        Second chromaticity coordinate.
    theta : Theta
        Parameter set.
    M : float
        Scaling factor (default 1.0).

    Returns
    -------
    lms : ndarray
        LMS coordinates, shape (..., 3).
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)

    # t = (γ + w_M*u1) / (1 - w_L*u1)
    t = (theta.gamma + theta.w_M * u1) / (1.0 - theta.w_L * u1)

    # D_t = w_L*t + w_M
    D_t = theta.w_L * t + theta.w_M

    # s = β*(t+1) + u2*D_t
    s = theta.beta * (t + 1.0) + u2 * D_t

    L = t * M
    S = s * M
    M_arr = np.full_like(L, M)

    return np.stack([L, M_arr, S], axis=-1)


def max_radius_in_direction(phi: float | np.ndarray, theta: Theta) -> np.ndarray:
    """Compute maximum attainable ||u|| in a given hue direction (closed-form).
    
    For direction d = (cos φ, sin φ), finds the maximum r such that
    u = r·d is in the attainable region.
    
    The constraints are linear in r, so the solution is closed-form:
    
    1. u₁ bounds: r·cos(φ) ∈ (-γ/w_M, 1/w_L)
       → r < bound / |cos(φ)| (depending on sign)
    
    2. u₂ > g(u₁): r·sin(φ) > g_intercept + g_slope·r·cos(φ)
       → r·(sin(φ) - g_slope·cos(φ)) > g_intercept
       This is also linear in r.
    
    Parameters
    ----------
    phi : float or array-like
        Hue angle(s) in radians.
    theta : Theta
        Parameter set.
        
    Returns
    -------
    r_max : ndarray
        Maximum attainable ||u|| for each direction.
    """
    phi = np.asarray(phi, dtype=float)
    scalar_input = (phi.ndim == 0)
    phi = np.atleast_1d(phi)
    
    lower, upper = u1_bounds(theta)
    
    # g(u1) = g_intercept + g_slope * u1
    Delta = theta.Delta
    g_intercept = -(theta.beta / Delta) * (theta.gamma + 1.0)
    g_slope = -(theta.beta / Delta) * (theta.w_M - theta.w_L)
    
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # Initialize with large value
    r_max = np.full_like(phi, 1e6, dtype=float)
    
    # Constraint 1: u1 bounds
    # If cos(φ) > 0: r < upper / cos(φ)
    # If cos(φ) < 0: r < lower / cos(φ) = |lower| / |cos(φ)|
    
    pos_cos = cos_phi > 1e-10
    neg_cos = cos_phi < -1e-10
    
    r_max = np.where(pos_cos, np.minimum(r_max, (upper - 1e-8) / cos_phi), r_max)
    r_max = np.where(neg_cos, np.minimum(r_max, (lower + 1e-8) / cos_phi), r_max)
    
    # Constraint 2: r·sin(φ) > g_intercept + g_slope·r·cos(φ)
    # Rearrange: r·(sin(φ) - g_slope·cos(φ)) > g_intercept
    coeff = sin_phi - g_slope * cos_phi
    
    # If coeff > 0: r > g_intercept / coeff (lower bound, doesn't constrain r_max)
    # If coeff < 0: r < g_intercept / coeff (upper bound)
    neg_coeff = coeff < -1e-10
    r_boundary = np.where(neg_coeff, g_intercept / coeff, 1e6)
    r_max = np.where(neg_coeff, np.minimum(r_max, r_boundary), r_max)
    
    # Ensure non-negative
    r_max = np.maximum(r_max, 0.0)
    
    if scalar_input:
        return r_max[0]
    return r_max


def attainable_v_boundary_polar(
    phi: float | np.ndarray,
    theta: Theta,
) -> np.ndarray:
    """Compute the v-space gamut boundary in polar coordinates (closed-form).
    
    For each hue angle φ, computes ||v||_max = tanh(κ·r_max(φ)).
    
    Parameters
    ----------
    phi : float or array-like
        Hue angle(s) in radians.
    theta : Theta
        Parameter set.
        
    Returns
    -------
    v_norm_max : ndarray
        Maximum ||v|| for each hue direction.
    """
    r_max_u = max_radius_in_direction(phi, theta)
    return np.tanh(theta.kappa * r_max_u)


__all__ = [
    "g_boundary",
    "u1_bounds",
    "in_attainable_region_u",
    "sample_attainable_region",
    "reconstruct_from_attainable",
    "max_radius_in_direction",
    "attainable_v_boundary_polar",
]
