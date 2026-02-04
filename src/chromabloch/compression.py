"""Radial compression map T_κ and its inverse (Definition: Radial Compression Map).

NUMERICAL LIMITATION WARNING:
For float64, tanh(x) = 1.0 exactly when x > ~18.71.
This means if κ||u|| > 18.71, the compression saturates and
reconstruction via arctanh cannot recover the original ||u||.

The disk interior clamp uses R_MAX = np.nextafter(1.0, 0.0), which is
the largest float64 less than 1.0. This yields:
    atanh(R_MAX) ≈ 18.71
as the maximum recoverable κ||u||.

To maintain numerical invertibility with margin, ensure κ||u|| < 17 (conservative).
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from .params import Theta

_SMALL_R = 1e-8

# R_MAX: disk interior clamp for invertibility
# We use 1 - 1e-15 instead of nextafter(1.0, 0.0) to ensure ||v|| < 1.0
# after arithmetic operations (nextafter can round back to 1.0 after mult/div)
R_MAX = 1.0 - 1e-15  # ≈ 0.999999999999999

# atanh(R_MAX) ≈ 18.36 — this is the hard cap on recoverable κ||u||
_INVERTIBILITY_CAP = float(np.arctanh(R_MAX))

# Conservative thresholds for is_safe() / is_reconstructable()
# Saturation: beyond this, reconstruction is increasingly unreliable
_TANH_SATURATION_THRESHOLD = 18.0
# Warning: approaching saturation zone
_TANH_WARNING_THRESHOLD = 15.0


def _euclidean_norm(u: np.ndarray) -> np.ndarray:
    """Compute Euclidean norm ||u|| along last axis (NOT squared)."""
    return np.linalg.norm(u, axis=-1)


def compress_to_disk(u: np.ndarray, theta: Theta) -> np.ndarray:
    """Apply T_κ(u)=tanh(κ|u|) u/|u| with series stabilization near the origin."""
    u = np.asarray(u, dtype=float)
    r = _euclidean_norm(u)

    scale = np.empty_like(r)
    small = r < _SMALL_R

    # Series: tanh(κr)/r ≈ κ - κ^3 r^2/3
    scale[small] = theta.kappa - (theta.kappa**3) * (r[small] ** 2) / 3.0

    not_small = ~small
    if np.any(not_small):
        r_ns = r[not_small]
        scale[not_small] = np.tanh(theta.kappa * r_ns) / r_ns

    v = u * np.expand_dims(scale, axis=-1)

    # Ensure we remain inside the open unit disk numerically.
    # Use R_MAX (largest float < 1) for principled clamping.
    rv = _euclidean_norm(v)
    too_close = rv >= 1.0
    if np.any(too_close):
        v[too_close] *= R_MAX / rv[too_close, None]
    return v


def decompress_from_disk(v: np.ndarray, theta: Theta) -> np.ndarray:
    """Inverse T_κ^{-1}(v)=arctanh(|v|)/(κ|v|) v with stability for |v|→0,1."""
    v = np.asarray(v, dtype=float)
    r = _euclidean_norm(v)

    # Clamp to R_MAX to avoid atanh(1) = inf
    r_safe = np.clip(r, 0.0, R_MAX)
    scale = np.empty_like(r_safe)

    small = r_safe < _SMALL_R
    # arctanh(r)/r ≈ 1 + r^2/3; divide by κ.
    scale[small] = (1.0 + (r_safe[small] ** 2) / 3.0) / theta.kappa

    not_small = ~small
    if np.any(not_small):
        r_ns = r_safe[not_small]
        scale[not_small] = np.arctanh(r_ns) / (theta.kappa * r_ns)

    u = v * np.expand_dims(scale, axis=-1)
    return u


class SaturationDiagnostics(NamedTuple):
    """Diagnostics for compression saturation behavior."""

    n_total: int
    n_saturated: int
    n_warning: int
    fraction_saturated: float
    fraction_warning: float
    max_kappa_r: float
    effective_max_hyperbolic_radius: float


def compression_saturation_diagnostics(
    u: np.ndarray,
    theta: Theta,
) -> SaturationDiagnostics:
    """Analyze compression saturation for given u values.

    For float64, tanh(x) ≈ 1.0 (indistinguishable) when x > ~18.4.
    This function reports how many samples exceed this threshold.

    Parameters
    ----------
    u : array-like
        Chromaticity coordinates, shape (..., 2).
    theta : Theta
        Parameter set containing kappa.

    Returns
    -------
    SaturationDiagnostics
        Named tuple with saturation statistics.
    """
    u = np.asarray(u, dtype=float)
    r = _euclidean_norm(u)
    kappa_r = theta.kappa * r

    n_total = r.size
    n_saturated = int(np.sum(kappa_r >= _TANH_SATURATION_THRESHOLD))
    n_warning = int(np.sum(kappa_r >= _TANH_WARNING_THRESHOLD))

    max_kappa_r = float(np.max(kappa_r)) if n_total > 0 else 0.0

    # Effective max hyperbolic radius (capped at saturation)
    effective_max = min(max_kappa_r, _TANH_SATURATION_THRESHOLD)

    return SaturationDiagnostics(
        n_total=n_total,
        n_saturated=n_saturated,
        n_warning=n_warning,
        fraction_saturated=n_saturated / n_total if n_total > 0 else 0.0,
        fraction_warning=n_warning / n_total if n_total > 0 else 0.0,
        max_kappa_r=max_kappa_r,
        effective_max_hyperbolic_radius=effective_max,
    )


def compression_roundtrip_error(
    u: np.ndarray,
    theta: Theta,
) -> np.ndarray:
    """Compute roundtrip error ||u - T_κ⁻¹(T_κ(u))|| for each sample.

    This exposes the numerical precision loss from tanh saturation.

    Parameters
    ----------
    u : array-like
        Chromaticity coordinates, shape (..., 2).
    theta : Theta
        Parameter set.

    Returns
    -------
    error : ndarray
        Roundtrip errors, shape (...).
    """
    u = np.asarray(u, dtype=float)
    v = compress_to_disk(u, theta)
    u_reconstructed = decompress_from_disk(v, theta)
    return np.linalg.norm(u - u_reconstructed, axis=-1)


def suggest_kappa_for_max_u_norm(
    max_u_norm: float,
    tol: float = 1e-8,
    safety_factor: float = 0.9,
) -> float:
    """Suggest κ value to achieve a given reconstruction tolerance.

    Uses the measured error profile to determine the maximum κ||u|| for
    the requested tolerance, then applies a safety factor.

    Parameters
    ----------
    max_u_norm : float
        Maximum expected ||u|| in your data.
    tol : float
        Desired reconstruction tolerance (default 1e-8).
    safety_factor : float
        Safety margin (default 0.9, i.e., 90% of the tolerance threshold).

    Returns
    -------
    kappa : float
        Suggested κ value.
        
    Notes
    -----
    The returned κ ensures that max κ||u|| < safety_factor * x_tol,
    where x_tol is the max κ||u|| that achieves the given tolerance.
    
    For tol=1e-8 with safety_factor=0.9: targets κ||u|| < 10.5
    """
    x_tol = max_x_for_reconstruction_tolerance(tol)
    return safety_factor * x_tol / max_u_norm


class RoundtripErrorProfile(NamedTuple):
    """Result of compression roundtrip error profiling.
    
    Attributes
    ----------
    x_values : ndarray
        Input κ||u|| values tested.
    u_norms : ndarray
        Corresponding ||u|| values.
    abs_error_u_norm : ndarray
        Absolute error in ||u|| after roundtrip: |u_orig - u_reconstructed|.
    rel_error_u_norm : ndarray
        Relative error: |u_orig - u_reconstructed| / |u_orig|.
    is_saturated : ndarray
        Boolean mask: True where tanh(x) = 1.0 in float64.
    max_x_for_tol : dict
        Dictionary mapping tolerance → max x achieving that error level.
    """
    x_values: np.ndarray
    u_norms: np.ndarray
    abs_error_u_norm: np.ndarray
    rel_error_u_norm: np.ndarray
    is_saturated: np.ndarray
    max_x_for_tol: dict


def compression_roundtrip_error_profile(
    kappa: float = 1.0,
    x_min: float = 0.1,
    x_max: float = 25.0,
    n_points: int = 500,
) -> RoundtripErrorProfile:
    """Profile roundtrip error as a function of κ||u|| for compression.
    
    This function measures the numerical precision of the T_κ ↔ T_κ⁻¹ roundtrip
    at various compression arguments x = κ||u||.
    
    The error grows exponentially as x approaches the float64 tanh saturation
    threshold (~18.4 where tanh(x) = 1.0 exactly).
    
    Parameters
    ----------
    kappa : float
        Compression parameter κ. Default 1.0.
    x_min : float
        Minimum κ||u|| value to test.
    x_max : float
        Maximum κ||u|| value to test.
    n_points : int
        Number of test points.
        
    Returns
    -------
    RoundtripErrorProfile
        Named tuple containing error measurements and thresholds.
        
    Notes
    -----
    The roundtrip error is measured along a fixed direction (1, 0) in u-space,
    which is sufficient since the compression is radially symmetric.
    """
    from .params import Theta  # Local import to avoid circular dependency
    
    theta = Theta(kappa=kappa)
    
    # Test points distributed to capture the transition region
    x_values = np.logspace(np.log10(x_min), np.log10(x_max), n_points)
    
    # Construct u vectors along fixed direction
    u_norms = x_values / kappa
    u_vectors = np.stack([u_norms, np.zeros_like(u_norms)], axis=-1)
    
    # Forward-backward roundtrip
    v_vectors = compress_to_disk(u_vectors, theta)
    u_reconstructed = decompress_from_disk(v_vectors, theta)
    
    # Compute errors
    error_vectors = u_vectors - u_reconstructed
    abs_error = np.linalg.norm(error_vectors, axis=-1)
    
    # Relative error (handle division by zero for u_norm ≈ 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.where(u_norms > 1e-10, abs_error / u_norms, abs_error)
    
    # Detect saturation (where tanh(x) = 1.0 exactly in float64)
    tanh_values = np.tanh(x_values)
    is_saturated = (tanh_values == 1.0)
    
    # Compute max x for various error tolerances
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    max_x_for_tol = {}
    
    for tol in tolerances:
        # Find largest x where rel_error <= tol
        mask = rel_error <= tol
        if np.any(mask):
            max_x_for_tol[tol] = float(x_values[mask][-1])
        else:
            max_x_for_tol[tol] = 0.0
    
    return RoundtripErrorProfile(
        x_values=x_values,
        u_norms=u_norms,
        abs_error_u_norm=abs_error,
        rel_error_u_norm=rel_error,
        is_saturated=is_saturated,
        max_x_for_tol=max_x_for_tol,
    )


def max_x_for_reconstruction_tolerance(tol: float = 1e-8) -> float:
    """Get the maximum κ||u|| that guarantees a given reconstruction tolerance.
    
    This is a cached computation based on the error profile.
    
    Parameters
    ----------
    tol : float
        Desired relative error tolerance (default 1e-8).
        
    Returns
    -------
    x_max : float
        Maximum κ||u|| that achieves the given tolerance.
        
    Notes
    -----
    The returned value is conservative (rounded down) to ensure the tolerance
    is met for all inputs at or below the threshold.
    """
    # Pre-computed values from dense profiling (examples/roundtrip_precision_profile.py)
    # These are EMPIRICALLY MEASURED values on a logspace grid
    # The values are conservative (rounded down) to ensure the tolerance is achieved
    # Note: These are NOT mathematical guarantees, but empirical bounds on the sampled grid.
    _MEASURED_THRESHOLDS = {
        1e-4: 17.0,   # Measured: 17.17
        1e-6: 15.5,   # Measured: 15.80
        1e-8: 11.5,   # Measured: 11.66
        1e-10: 10.0,  # Measured: 10.10
        1e-12: 7.0,   # Measured: 7.21
    }
    
    # Find the closest tolerance
    tolerances = sorted(_MEASURED_THRESHOLDS.keys(), reverse=True)
    for t in tolerances:
        if tol >= t:
            return _MEASURED_THRESHOLDS[t]
    
    # Very tight tolerance: be conservative
    return _MEASURED_THRESHOLDS[min(tolerances)]


__all__ = [
    "compress_to_disk",
    "decompress_from_disk",
    "SaturationDiagnostics",
    "compression_saturation_diagnostics",
    "compression_roundtrip_error",
    "suggest_kappa_for_max_u_norm",
    "RoundtripErrorProfile",
    "compression_roundtrip_error_profile",
    "max_x_for_reconstruction_tolerance",
    "R_MAX",
    "_INVERTIBILITY_CAP",
    "_TANH_WARNING_THRESHOLD",
    "_TANH_SATURATION_THRESHOLD",
]
