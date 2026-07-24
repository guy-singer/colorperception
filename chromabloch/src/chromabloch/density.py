"""Density matrix representation ρ(v) and chromatic attributes.

For v = (v1, v2) ∈ D̄ (closed disk), the rebit density matrix is:
    ρ(v) = ½(I₂ + v₁σ₁ + v₂σ₂)

where σ₁ = diag(1,-1) and σ₂ = [[0,1],[1,0]] are the real Pauli matrices.

References:
    Definition: Density Matrix Representation
    Definition: Von Neumann Entropy
    Definition: Information-Theoretic Saturation
    Definition: Hue
"""

from __future__ import annotations

import numpy as np

# Numerical tolerance for validation
_TOL = 1e-9


def rho_of_v(v: np.ndarray) -> np.ndarray:
    """Construct the 2×2 density matrix ρ(v) from Bloch coordinates.

    Definition: Density Matrix Representation.

    ρ(v) = ½ [[1+v1, v2], [v2, 1-v1]]

    Parameters
    ----------
    v : array-like
        Bloch disk coordinates, shape (..., 2).

    Returns
    -------
    rho : ndarray
        Density matrices, shape (..., 2, 2).
    """
    v = np.asarray(v, dtype=float)
    v1 = v[..., 0]
    v2 = v[..., 1]

    rho = np.empty(v.shape[:-1] + (2, 2), dtype=float)
    rho[..., 0, 0] = 0.5 * (1.0 + v1)
    rho[..., 0, 1] = 0.5 * v2
    rho[..., 1, 0] = 0.5 * v2
    rho[..., 1, 1] = 0.5 * (1.0 - v1)
    return rho


def bloch_from_rho(rho: np.ndarray, strict: bool = False) -> np.ndarray:
    """Recover Bloch coordinates v from a density matrix ρ.

    Proposition: Density Matrix Properties (converse).

    Given ρ = [[a, b], [b, c]] with trace 1 and ρ ⪰ 0:
        v1 = a - c = 2a - 1
        v2 = 2b

    Parameters
    ----------
    rho : array-like
        Density matrices, shape (..., 2, 2).
    strict : bool
        If True, validate PSD and trace=1 constraints.

    Returns
    -------
    v : ndarray
        Bloch coordinates, shape (..., 2).
    """
    rho = np.asarray(rho, dtype=float)
    if strict:
        if not np.allclose(trace_is_one(rho), True):
            raise ValueError("Density matrix does not have unit trace")
        if not np.all(is_psd_2x2(rho)):
            raise ValueError("Density matrix is not positive semidefinite")

    a = rho[..., 0, 0]
    b = rho[..., 0, 1]
    c = rho[..., 1, 1]

    v1 = a - c  # equivalently: 2*a - 1
    v2 = 2.0 * b
    return np.stack([v1, v2], axis=-1)


def is_psd_2x2(rho: np.ndarray, tol: float = _TOL) -> np.ndarray:
    """Check if 2×2 symmetric matrices are positive semidefinite.

    For symmetric 2×2: PSD ⟺ trace ≥ 0 and det ≥ 0.

    Parameters
    ----------
    rho : array-like
        Matrices, shape (..., 2, 2).
    tol : float
        Tolerance for eigenvalue checks.

    Returns
    -------
    is_psd : ndarray of bool
        Shape (...).
    """
    rho = np.asarray(rho, dtype=float)
    a = rho[..., 0, 0]
    b = rho[..., 0, 1]
    c = rho[..., 1, 1]
    tr = a + c
    det = a * c - b * b
    return (tr >= -tol) & (det >= -tol)


def trace_is_one(rho: np.ndarray, tol: float = _TOL) -> np.ndarray:
    """Check if matrices have unit trace.

    Parameters
    ----------
    rho : array-like
        Matrices, shape (..., 2, 2).
    tol : float
        Tolerance.

    Returns
    -------
    result : ndarray of bool
        Shape (...).
    """
    rho = np.asarray(rho, dtype=float)
    tr = rho[..., 0, 0] + rho[..., 1, 1]
    return np.abs(tr - 1.0) < tol


def bloch_norm(v: np.ndarray) -> np.ndarray:
    """Compute ||v|| for Bloch coordinates.

    Parameters
    ----------
    v : array-like
        Bloch coordinates, shape (..., 2).

    Returns
    -------
    r : ndarray
        Norms, shape (...).
    """
    v = np.asarray(v, dtype=float)
    return np.linalg.norm(v, axis=-1)


def von_neumann_entropy(v: np.ndarray, base: float = 2.0) -> np.ndarray:
    """Compute von Neumann entropy S(ρ) = -Tr(ρ log_base ρ).

    Definition: Von Neumann Entropy.

    Eigenvalues of ρ(v) are λ± = (1 ± ||v||)/2.
    S = -λ₊ log_base(λ₊) - λ₋ log_base(λ₋)

    Base conversion: S_b = S_2 / log_2(b) = S_e / ln(b)

    Parameters
    ----------
    v : array-like
        Bloch coordinates, shape (..., 2).
    base : float
        Logarithm base (default 2 for bits). Must be > 0 and != 1.

    Returns
    -------
    S : ndarray
        Entropy values, shape (...).
        For base=2: S(0)=1 bit (max), S(1)=0 bits (min).
        For base=e: S(0)=ln(2)≈0.693 nats, S(1)=0 nats.

    Raises
    ------
    ValueError
        If base <= 0 or base == 1.
    """
    if base <= 0:
        raise ValueError(f"Logarithm base must be positive, got {base}")
    if base == 1.0:
        raise ValueError("Logarithm base cannot be 1 (log_1 is undefined)")

    v = np.asarray(v, dtype=float)
    r = np.linalg.norm(v, axis=-1)

    # Clamp r to [0, 1-eps] for numerical stability
    r = np.clip(r, 0.0, 1.0 - 1e-15)

    # Eigenvalues: λ± = (1 ± r) / 2
    lam_plus = (1.0 + r) / 2.0
    lam_minus = (1.0 - r) / 2.0

    # Compute x*log(x) safely (0*log(0) = 0 by convention)
    def xlogx(x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        mask = x > 0
        result[mask] = x[mask] * np.log(x[mask])
        return result

    # S = -[λ₊ ln(λ₊) + λ₋ ln(λ₋)] / ln(base)
    # This is the correct formula that scales properly with base
    log_base = np.log(base)
    entropy = -(xlogx(lam_plus) + xlogx(lam_minus)) / log_base
    return entropy


def saturation_sigma(v: np.ndarray) -> np.ndarray:
    """Compute information-theoretic saturation Σ(r) = 1 - S(r).

    Definition: Information-Theoretic Saturation.

    Σ(0) = 0 (achromatic), Σ(1) = 1 (maximally saturated).

    Parameters
    ----------
    v : array-like
        Bloch coordinates, shape (..., 2).

    Returns
    -------
    sigma : ndarray
        Saturation values, shape (...).
    """
    return 1.0 - von_neumann_entropy(v, base=2.0)


def hue_angle(v: np.ndarray) -> np.ndarray:
    """Compute hue angle H(v) = atan2(v2, v1).

    Definition: Hue.

    Parameters
    ----------
    v : array-like
        Bloch coordinates, shape (..., 2).

    Returns
    -------
    H : ndarray
        Hue angles in radians (-π, π], shape (...).
        NaN for v = (0, 0).
    """
    v = np.asarray(v, dtype=float)
    v1 = v[..., 0]
    v2 = v[..., 1]

    # atan2 returns NaN for (0,0) naturally in some implementations,
    # but we make it explicit
    h = np.arctan2(v2, v1)

    # Mark achromatic points as NaN
    achromatic = (v1 == 0) & (v2 == 0)
    h = np.where(achromatic, np.nan, h)
    return h


__all__ = [
    "rho_of_v",
    "bloch_from_rho",
    "is_psd_2x2",
    "trace_is_one",
    "bloch_norm",
    "von_neumann_entropy",
    "saturation_sigma",
    "hue_angle",
]
