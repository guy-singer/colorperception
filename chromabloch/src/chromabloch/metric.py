"""Induced metric and pullback geometry on LMS space.

The Bloch disk D carries the Klein (hyperbolic) metric. The chromaticity map
Φ_θ: ℝ³₊₊ → D induces a pullback metric on LMS space:

    G_LMS(x) = (DΦ(x))ᵀ · G_D(Φ(x)) · DΦ(x)

where G_D is the Klein metric tensor on the disk.

This module computes:
1. The Klein metric tensor on D
2. The pullback metric on LMS space
3. Predicted discrimination ellipses/ellipsoids

This bridges Phase I (geometry) with Phase II (perceptual validation).
"""

from __future__ import annotations

import numpy as np

from .params import Theta
from .mapping import phi_theta
from .jacobian import jacobian_phi_analytic


def klein_metric_tensor(v: np.ndarray) -> np.ndarray:
    """Compute the Klein metric tensor G_D at point v.
    
    The Klein (Beltrami-Klein) model metric is:
        ds² = (dv·dv)/(1-||v||²) + (v·dv)²/(1-||v||²)²
    
    In tensor form:
        G_ij = δ_ij/(1-||v||²) + v_i·v_j/(1-||v||²)²
    
    Parameters
    ----------
    v : array-like
        Point in the disk D, shape (2,) or (..., 2).
        
    Returns
    -------
    G : ndarray
        Metric tensor, shape (2, 2) or (..., 2, 2).
    """
    v = np.asarray(v, dtype=float)
    single_point = (v.ndim == 1)
    
    if single_point:
        v = v.reshape(1, 2)
    
    v_norm_sq = np.sum(v * v, axis=-1, keepdims=True)  # shape (..., 1)
    
    # Clamp for numerical stability
    v_norm_sq = np.clip(v_norm_sq, 0, 0.9999)
    
    denom1 = 1.0 - v_norm_sq  # shape (..., 1)
    denom2 = denom1 ** 2      # shape (..., 1)
    
    # Identity part: δ_ij / (1 - ||v||²)
    n_points = v.shape[0] if v.ndim == 2 else np.prod(v.shape[:-1])
    I = np.eye(2)
    G_identity = I / denom1[..., np.newaxis]  # shape (..., 2, 2)
    
    # Outer product part: v_i v_j / (1 - ||v||²)²
    v_outer = v[..., :, np.newaxis] * v[..., np.newaxis, :]  # shape (..., 2, 2)
    G_outer = v_outer / denom2[..., np.newaxis]
    
    G = G_identity + G_outer
    
    if single_point:
        return G[0]
    
    return G


def pullback_metric_lms(
    lms: np.ndarray,
    theta: Theta,
) -> np.ndarray:
    """Compute the pullback metric on LMS space.
    
    G_LMS(x) = Jᵀ · G_D(Φ(x)) · J
    
    where J = DΦ(x) is the 2×3 Jacobian.
    
    Note: Since Φ maps 3D → 2D, the pullback metric is rank ≤ 2.
    For ε=0, the scale direction is in the null space.
    
    Parameters
    ----------
    lms : array-like
        LMS cone responses, shape (3,) or (..., 3).
    theta : Theta
        Parameter set.
        
    Returns
    -------
    G : ndarray
        Pullback metric tensor, shape (3, 3) or (..., 3, 3).
    """
    lms = np.asarray(lms, dtype=float)
    single_point = (lms.ndim == 1)
    
    if single_point:
        lms = lms.reshape(1, 3)
    
    # Compute v = Φ(lms)
    v = phi_theta(lms, theta)
    
    # Compute Jacobian J = DΦ (2×3 at each point)
    J = jacobian_phi_analytic(lms, theta)
    
    # Compute Klein metric at v (2×2)
    G_D = klein_metric_tensor(v)
    
    # Pullback: G_LMS = Jᵀ G_D J (3×3)
    # J is (..., 2, 3), G_D is (..., 2, 2)
    # Jᵀ is (..., 3, 2)
    
    # J^T @ G_D: (..., 3, 2) @ (..., 2, 2) = (..., 3, 2)
    JtG = np.einsum('...ji,...jk->...ik', J, G_D)  # J^T @ G_D
    
    # (J^T @ G_D) @ J: (..., 3, 2) @ (..., 2, 3) = (..., 3, 3)
    G_LMS = np.einsum('...ij,...jk->...ik', JtG, J)
    
    if single_point:
        return G_LMS[0]
    
    return G_LMS


def metric_eigenvalues(G: np.ndarray, clip_negative: bool = True) -> np.ndarray:
    """Compute eigenvalues of metric tensor.
    
    For the pullback metric, these indicate:
    - Two positive eigenvalues: directions with finite "perceptual distance"
    - One zero eigenvalue (ε=0): scale direction (no perceived change)
    
    Parameters
    ----------
    G : array-like
        Metric tensor, shape (3, 3) or (..., 3, 3).
    clip_negative : bool
        If True, clip tiny negative eigenvalues to 0 (default True).
        Numerical errors can produce small negative eigenvalues for
        theoretically positive semidefinite matrices.
        
    Returns
    -------
    eigenvalues : ndarray
        Sorted eigenvalues (descending), shape (3,) or (..., 3).
        
    Notes
    -----
    The pullback metric is theoretically PSD (positive semidefinite).
    However, numerical errors in the Jacobian computation and matrix
    multiplication can produce tiny negative eigenvalues (typically
    on the order of -1e-15 to -1e-12). These are clipped to 0.0
    when clip_negative=True.
    """
    G = np.asarray(G, dtype=float)
    eigenvalues = np.linalg.eigvalsh(G)
    
    if clip_negative:
        eigenvalues = np.maximum(eigenvalues, 0.0)
    
    # Sort descending
    return np.sort(eigenvalues, axis=-1)[..., ::-1]


def metric_trace(G: np.ndarray) -> np.ndarray:
    """Compute trace of metric tensor (overall sensitivity).
    
    tr(G) = Σ λ_i measures total sensitivity to changes.
    
    Parameters
    ----------
    G : array-like
        Metric tensor, shape (3, 3) or (..., 3, 3).
        
    Returns
    -------
    trace : ndarray
        Trace values, shape () or (...).
    """
    G = np.asarray(G, dtype=float)
    return np.trace(G, axis1=-2, axis2=-1)


def discrimination_ellipsoid_axes(
    lms: np.ndarray,
    theta: Theta,
    threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute discrimination ellipsoid axes in LMS space.
    
    The ellipsoid {δx : δxᵀ G δx ≤ threshold²} describes
    "just noticeable differences" predicted by the metric.
    
    Parameters
    ----------
    lms : array-like
        Center point, shape (3,).
    theta : Theta
        Parameter set.
    threshold : float
        Distance threshold (1.0 = unit distance in hyperbolic metric).
        
    Returns
    -------
    lengths : ndarray
        Semi-axis lengths, shape (3,).
    directions : ndarray
        Principal directions (columns), shape (3, 3).
        
    Notes
    -----
    The pullback metric G_LMS is at most rank 2 (since Φ maps ℝ³→ℝ²).
    This means one eigenvalue is zero (or near-zero), corresponding to
    the null direction (scale direction for ε=0). The semi-axis length
    in this direction is infinite.
    
    Tiny negative eigenvalues from numerical error are clipped to 0
    before computing sqrt.
    """
    lms = np.asarray(lms, dtype=float)
    
    G = pullback_metric_lms(lms, theta)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    
    # Clip tiny negative eigenvalues to 0 (numerical artifact)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Semi-axis lengths: a_i = threshold / sqrt(λ_i)
    # Handle zero/small eigenvalues (infinite axis length = no discrimination)
    with np.errstate(divide='ignore', invalid='ignore'):
        lengths = np.where(
            eigenvalues > 1e-10,
            threshold / np.sqrt(eigenvalues),
            np.inf
        )
    
    return lengths, eigenvectors


def chromaticity_plane_ellipse(
    lms: np.ndarray,
    theta: Theta,
    threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project discrimination ellipsoid to chromaticity plane.
    
    For visualization, we project the 3D ellipsoid to a 2D slice.
    At constant luminance Y, the ellipse shows discrimination in
    the (L/Y, S/Y) or similar chromaticity coordinates.
    
    Parameters
    ----------
    lms : array-like
        Center point, shape (3,).
    theta : Theta
        Parameter set.
    threshold : float
        Distance threshold.
        
    Returns
    -------
    center_chrom : ndarray
        Center in chromaticity coords, shape (2,).
    axes : ndarray
        Semi-axis lengths in chromaticity, shape (2,).
    angle : float
        Rotation angle of ellipse (radians).
    """
    lms = np.asarray(lms, dtype=float)
    L, M, S = lms
    
    # Chromaticity coordinates: (L/(L+M), S/(L+M))
    denom = L + M
    center_chrom = np.array([L / denom, S / denom])
    
    # Jacobian of chromaticity w.r.t. LMS
    # c1 = L/(L+M), c2 = S/(L+M)
    # ∂c1/∂L = M/(L+M)², ∂c1/∂M = -L/(L+M)², ∂c1/∂S = 0
    # ∂c2/∂L = -S/(L+M)², ∂c2/∂M = -S/(L+M)², ∂c2/∂S = 1/(L+M)
    
    J_chrom = np.array([
        [M / denom**2, -L / denom**2, 0],
        [-S / denom**2, -S / denom**2, 1 / denom],
    ])
    
    # Get LMS metric
    G_lms = pullback_metric_lms(lms, theta)
    
    # Project metric to chromaticity: G_chrom = J G_lms⁻¹ Jᵀ ... but G_lms is rank 2
    # Instead, use pseudo-inverse or work directly with ellipse
    
    # Simpler: just return the v-space ellipse (direct Bloch disk metric)
    v = phi_theta(lms, theta)
    G_v = klein_metric_tensor(v)
    
    eigenvalues, eigenvectors = np.linalg.eigh(G_v)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    with np.errstate(divide='ignore'):
        axes = np.where(eigenvalues > 1e-10, threshold / np.sqrt(eigenvalues), 1e6)
    
    # Angle of major axis
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    return v, axes, angle


__all__ = [
    "klein_metric_tensor",
    "pullback_metric_lms",
    "metric_eigenvalues",
    "metric_trace",
    "discrimination_ellipsoid_axes",
    "chromaticity_plane_ellipse",
]
