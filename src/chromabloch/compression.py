"""Radial compression map T_κ and its inverse (Definition: Radial Compression Map).

NUMERICAL LIMITATION WARNING:
For float64, tanh(x) ≈ 1.0 when x > ~18.4.
This means if κ||u|| > 18.4, the compression effectively saturates and
reconstruction via arctanh cannot recover the original ||u||.

To maintain numerical invertibility, ensure κ||u|| < 15 (conservative).
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from .params import Theta

_SMALL_R = 1e-8
_BOUND_MARGIN = 1e-12
# Threshold beyond which tanh saturates in float64
_TANH_SATURATION_THRESHOLD = 18.0
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
    rv = _euclidean_norm(v)
    too_close = rv >= 1.0
    if np.any(too_close):
        v[too_close] *= (1.0 - _BOUND_MARGIN) / rv[too_close, None]
    return v


def decompress_from_disk(v: np.ndarray, theta: Theta) -> np.ndarray:
    """Inverse T_κ^{-1}(v)=arctanh(|v|)/(κ|v|) v with stability for |v|→0,1."""
    v = np.asarray(v, dtype=float)
    r = _euclidean_norm(v)

    r_safe = np.clip(r, 0.0, 1.0 - _BOUND_MARGIN)
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
    safety_factor: float = 0.8,
) -> float:
    """Suggest κ value to avoid saturation for given max ||u||.

    To maintain numerical invertibility, we want κ||u|| < 15.
    This function returns κ ≤ safety_factor * 15 / max_u_norm.

    Parameters
    ----------
    max_u_norm : float
        Maximum expected ||u|| in your data.
    safety_factor : float
        Safety margin (default 0.8).

    Returns
    -------
    kappa : float
        Suggested κ value.
    """
    return safety_factor * _TANH_WARNING_THRESHOLD / max_u_norm


__all__ = [
    "compress_to_disk",
    "decompress_from_disk",
    "SaturationDiagnostics",
    "compression_saturation_diagnostics",
    "compression_roundtrip_error",
    "suggest_kappa_for_max_u_norm",
]
