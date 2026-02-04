"""Chromaticity projection Π and full map Φ_θ (Definition: Chromaticity Map).

The chromaticity map Φ_θ: L → D is the composition:
    Φ_θ = T_κ ∘ Π ∘ O

where:
    O: opponent transform (L,M,S) → (Y, O1, O2)
    Π: chromaticity projection (Y, O1, O2) → u = (u1, u2)
    T_κ: radial compression u → v ∈ D

Domain Contracts:
    Mathematical: LMS ∈ ℝ³₊₊ (strictly positive)
    Numerical (ε > 0): LMS ∈ ℝ³₊ (nonnegative) is safe
"""

from __future__ import annotations

from typing import NamedTuple, Optional
import warnings

import numpy as np

from .params import Theta
from .opponent import opponent_transform
from .compression import compress_to_disk, _TANH_SATURATION_THRESHOLD, _TANH_WARNING_THRESHOLD


class DomainViolation(ValueError):
    """Raised when strict domain checks fail."""
    pass


class MappingDiagnostics(NamedTuple):
    """Comprehensive diagnostics from the mapping pipeline.
    
    Attributes
    ----------
    n_total : int
        Total number of input samples.
    n_negative_clipped : int
        Number of negative LMS values clipped to zero.
    n_zero_lms : int
        Number of exact zero LMS values (per channel).
    min_Y : float
        Minimum luminance value.
    max_Y : float
        Maximum luminance value.
    min_u_norm : float
        Minimum ||u|| (chromaticity magnitude).
    max_u_norm : float
        Maximum ||u|| (chromaticity magnitude).
    max_kappa_u : float
        Maximum κ||u|| (compression argument).
    n_near_saturation : int
        Number of samples with κ||u|| > warning threshold (15).
    n_saturated : int
        Number of samples with κ||u|| > saturation threshold (18).
    n_boundary_clamped : int
        Number of outputs clamped to ||v|| < 1.
    max_v_norm_unclamped : float
        Maximum ||v|| before clamping (shows true saturation).
    """
    n_total: int
    n_negative_clipped: int
    n_zero_lms: int
    min_Y: float
    max_Y: float
    min_u_norm: float
    max_u_norm: float
    max_kappa_u: float
    n_near_saturation: int
    n_saturated: int
    n_boundary_clamped: int
    max_v_norm_unclamped: float
    
    def is_safe(self) -> bool:
        """Return True if no saturation or domain issues detected.
        
        Checks:
        - No saturation (κ||u|| > 18)
        - No boundary clamping
        - No negative clipping
        - max_kappa_u < 15 (conservative threshold for reconstruction reliability)
        """
        return (self.n_saturated == 0 and 
                self.n_boundary_clamped == 0 and
                self.n_negative_clipped == 0 and
                self.max_kappa_u < _TANH_WARNING_THRESHOLD)
    
    def is_reconstructable(self) -> bool:
        """Return True if reconstruction is guaranteed to be accurate.
        
        More conservative than is_safe(): requires max_kappa_u < 14.5
        to ensure reconstruction error < 1e-8.
        """
        return (self.is_safe() and self.max_kappa_u < 14.5)
    
    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Mapping Diagnostics ({self.n_total} samples)",
            f"  Domain: {self.n_negative_clipped} clipped, {self.n_zero_lms} zeros",
            f"  Luminance Y: [{self.min_Y:.4g}, {self.max_Y:.4g}]",
            f"  Chromaticity ||u||: [{self.min_u_norm:.4g}, {self.max_u_norm:.4g}]",
            f"  Compression κ||u|| max: {self.max_kappa_u:.4g}",
            f"  Saturation: {self.n_near_saturation} near, {self.n_saturated} saturated",
            f"  Boundary: {self.n_boundary_clamped} clamped, max ||v|| = {self.max_v_norm_unclamped:.6f}",
        ]
        return "\n".join(lines)


def chromaticity_projection(
    Y: np.ndarray,
    O1: np.ndarray,
    O2: np.ndarray,
    theta: Theta,
) -> np.ndarray:
    """Project opponent coordinates to chromaticity u = (O1, O2) / (Y + ε).

    Definition: Chromaticity Projection (Π).

    Parameters
    ----------
    Y : array-like
        Luminance values, shape (...).
    O1 : array-like
        First opponent coordinate, shape (...).
    O2 : array-like
        Second opponent coordinate, shape (...).
    theta : Theta
        Parameter set containing epsilon.

    Returns
    -------
    u : ndarray
        Chromaticity coordinates, shape (..., 2).
    """
    Y = np.asarray(Y, dtype=float)
    O1 = np.asarray(O1, dtype=float)
    O2 = np.asarray(O2, dtype=float)

    denom = Y + theta.epsilon
    u1 = O1 / denom
    u2 = O2 / denom
    return np.stack([u1, u2], axis=-1)


def _validate_lms_strict(lms: np.ndarray, theta: Theta) -> None:
    """Validate LMS values for strict mathematical domain."""
    if np.any(lms <= 0):
        n_nonpositive = np.sum(lms <= 0)
        raise DomainViolation(
            f"Strict domain requires LMS > 0, but {n_nonpositive} values are ≤ 0. "
            f"Use strict_domain=False for image processing with zeros."
        )
    _validate_luminance(lms, theta)


def _validate_luminance(lms: np.ndarray, theta: Theta) -> None:
    """Validate that Y + ε > 0 to avoid division by zero."""
    Y = theta.w_L * lms[..., 0] + theta.w_M * lms[..., 1]
    denom = Y + theta.epsilon
    if np.any(denom <= 0):
        n_bad = np.sum(denom <= 0)
        raise DomainViolation(
            f"Division by zero: Y + ε ≤ 0 for {n_bad} samples. "
            f"With ε={theta.epsilon}, this requires Y > {-theta.epsilon}. "
            f"Use ε > 0 for inputs with zero luminance (black pixels)."
        )


def phi_theta(
    lms: np.ndarray, 
    theta: Theta,
    strict_domain: bool = False,
) -> np.ndarray:
    """Compute the full chromaticity map Φ_θ(LMS) → v ∈ D.

    Definition: Chromaticity Map (Φ_θ = T_κ ∘ Π ∘ O).

    Parameters
    ----------
    lms : array-like
        LMS cone responses, shape (..., 3).
    theta : Theta
        Parameter set.
    strict_domain : bool, default False
        If True, raise DomainViolation for non-positive LMS.
        If False, clip negatives to 0 with warning (image processing mode).

    Returns
    -------
    v : ndarray
        Bloch disk coordinates, shape (..., 2), with ||v|| < 1.
        
    Raises
    ------
    DomainViolation
        If strict_domain=True and LMS contains non-positive values.
    """
    lms = np.asarray(lms, dtype=float)
    
    if strict_domain:
        _validate_lms_strict(lms, theta)
    else:
        # Image processing mode: clip negatives
        n_neg = np.sum(lms < 0)
        if n_neg > 0:
            warnings.warn(
                f"Clipping {n_neg} negative LMS values to 0 (use strict_domain=True to raise)",
                stacklevel=2,
            )
            lms = np.maximum(lms, 0.0)
        # Even in image mode, must validate Y + ε > 0
        _validate_luminance(lms, theta)
    
    Y, O1, O2 = opponent_transform(lms, theta)
    u = chromaticity_projection(Y, O1, O2, theta)
    v = compress_to_disk(u, theta)
    return v


def phi_theta_with_diagnostics(
    lms: np.ndarray,
    theta: Theta,
    strict_domain: bool = False,
) -> tuple[np.ndarray, MappingDiagnostics]:
    """Compute Φ_θ with comprehensive diagnostics.
    
    Parameters
    ----------
    lms : array-like
        LMS cone responses, shape (..., 3).
    theta : Theta
        Parameter set.
    strict_domain : bool, default False
        If True, raise on non-positive LMS.
        
    Returns
    -------
    v : ndarray
        Bloch disk coordinates, shape (..., 2).
    diag : MappingDiagnostics
        Comprehensive diagnostics about the mapping.
    """
    lms = np.asarray(lms, dtype=float)
    original_shape = lms.shape
    lms_flat = lms.reshape(-1, 3)
    n_total = lms_flat.shape[0]
    
    # Domain diagnostics
    n_negative_clipped = int(np.sum(lms_flat < 0))
    n_zero_lms = int(np.sum(lms_flat == 0))
    
    if strict_domain:
        _validate_lms_strict(lms_flat.reshape(original_shape), theta)
    else:
        lms_flat = np.maximum(lms_flat, 0.0)
        # Even in image mode, must validate Y + ε > 0
        _validate_luminance(lms_flat, theta)
    
    # Forward pass
    Y, O1, O2 = opponent_transform(lms_flat, theta)
    u = chromaticity_projection(Y, O1, O2, theta)
    
    # Chromaticity diagnostics
    u_norm = np.linalg.norm(u, axis=-1)
    kappa_u = theta.kappa * u_norm
    
    # Pre-compression analysis
    n_near_saturation = int(np.sum(kappa_u > _TANH_WARNING_THRESHOLD))
    n_saturated = int(np.sum(kappa_u > _TANH_SATURATION_THRESHOLD))
    
    # Compute v and track clamping
    tanh_val = np.tanh(kappa_u)
    v_norm_unclamped = tanh_val.copy()
    
    # Compression with direction
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.where(u_norm > 1e-10, tanh_val / u_norm, theta.kappa)
    v = u * scale[..., np.newaxis]
    
    # Check for clamping
    v_norm = np.linalg.norm(v, axis=-1)
    n_boundary_clamped = int(np.sum(v_norm >= 1.0 - 1e-12))
    
    # Clamp to open disk
    mask = v_norm >= 1.0
    if np.any(mask):
        v[mask] *= (1.0 - 1e-12) / v_norm[mask, np.newaxis]
    
    # Reshape back
    v = v.reshape(original_shape[:-1] + (2,))
    
    diag = MappingDiagnostics(
        n_total=n_total,
        n_negative_clipped=n_negative_clipped,
        n_zero_lms=n_zero_lms,
        min_Y=float(np.min(Y)),
        max_Y=float(np.max(Y)),
        min_u_norm=float(np.min(u_norm)),
        max_u_norm=float(np.max(u_norm)),
        max_kappa_u=float(np.max(kappa_u)),
        n_near_saturation=n_near_saturation,
        n_saturated=n_saturated,
        n_boundary_clamped=n_boundary_clamped,
        max_v_norm_unclamped=float(np.max(v_norm_unclamped)),
    )
    
    return v, diag


class PhiComponents(NamedTuple):
    """Intermediate values from the chromaticity map pipeline."""

    Y: np.ndarray
    O1: np.ndarray
    O2: np.ndarray
    u: np.ndarray
    v: np.ndarray


def phi_theta_components(lms: np.ndarray, theta: Theta) -> PhiComponents:
    """Compute Φ_θ returning all intermediate values for debugging.

    Parameters
    ----------
    lms : array-like
        LMS cone responses, shape (..., 3).
    theta : Theta
        Parameter set.

    Returns
    -------
    PhiComponents
        Named tuple with (Y, O1, O2, u, v).
    """
    lms = np.asarray(lms, dtype=float)
    Y, O1, O2 = opponent_transform(lms, theta)
    u = chromaticity_projection(Y, O1, O2, theta)
    v = compress_to_disk(u, theta)
    return PhiComponents(Y=Y, O1=O1, O2=O2, u=u, v=v)


__all__ = [
    "chromaticity_projection", 
    "phi_theta", 
    "phi_theta_with_diagnostics",
    "phi_theta_components", 
    "PhiComponents",
    "MappingDiagnostics",
    "DomainViolation",
]
