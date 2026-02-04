"""Quantum-information-theoretic distances on the Bloch disk.

Since ρ(v) is a valid 2×2 density matrix (positive semidefinite, trace 1),
we can compute various quantum distances/divergences:

1. Trace distance: D_tr(ρ, σ) = ½||ρ - σ||₁
2. Fidelity: F(ρ, σ) = (tr√(√ρ σ √ρ))²
3. Bures distance: D_B(ρ, σ) = √(2(1 - √F))
4. Fubini-Study distance (pure states): d_FS = arccos(√F)

These can be compared to the Hilbert distance to understand
which quantum notion best captures "perceptual distance."
"""

from __future__ import annotations

import numpy as np

from .density import rho_of_v
from .compression import R_MAX


def trace_distance(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute trace distance between density matrices.
    
    D_tr(ρ₁, ρ₂) = ½ ||ρ₁ - ρ₂||₁ = ½ tr|ρ₁ - ρ₂|
    
    For 2×2 density matrices (qubits/rebits), this simplifies to:
        D_tr = ½ ||v₁ - v₂||
    
    Parameters
    ----------
    v1 : array-like
        First Bloch vector, shape (2,) or (..., 2).
    v2 : array-like
        Second Bloch vector, shape (2,) or (..., 2).
        
    Returns
    -------
    d : ndarray
        Trace distance(s).
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    
    # For qubits: D_tr = ½ ||v1 - v2||
    diff = v1 - v2
    return 0.5 * np.linalg.norm(diff, axis=-1)


def fidelity(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute fidelity between density matrices.
    
    F(ρ₁, ρ₂) = (tr√(√ρ₁ ρ₂ √ρ₁))²
    
    For 2×2 density matrices, this simplifies to:
        F = (tr√(ρ₁ρ₂ + √(det(ρ₁)det(ρ₂)) I))²
    
    For rebit states (real density matrices), further simplification:
        F = ½(1 + v₁·v₂ + √((1-||v₁||²)(1-||v₂||²)))
    
    Parameters
    ----------
    v1 : array-like
        First Bloch vector, shape (2,) or (..., 2).
    v2 : array-like
        Second Bloch vector, shape (2,) or (..., 2).
        
    Returns
    -------
    F : ndarray
        Fidelity values in [0, 1].
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    
    r1_sq = np.sum(v1 * v1, axis=-1)
    r2_sq = np.sum(v2 * v2, axis=-1)
    v1_dot_v2 = np.sum(v1 * v2, axis=-1)
    
    # Clamp for numerical stability (ensure 1 - r² > 0)
    r_sq_max = R_MAX ** 2
    r1_sq = np.clip(r1_sq, 0, r_sq_max)
    r2_sq = np.clip(r2_sq, 0, r_sq_max)
    
    # F = ½(1 + v₁·v₂ + √((1-r₁²)(1-r₂²)))
    sqrt_term = np.sqrt((1 - r1_sq) * (1 - r2_sq))
    F = 0.5 * (1 + v1_dot_v2 + sqrt_term)
    
    # Clamp to [0, 1]
    return np.clip(F, 0, 1)


def bures_distance(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute Bures distance between density matrices.
    
    D_B(ρ₁, ρ₂) = √(2(1 - √F(ρ₁, ρ₂)))
    
    The Bures distance is a metric on density matrices,
    related to the Riemannian geometry of quantum states.
    
    Parameters
    ----------
    v1 : array-like
        First Bloch vector, shape (2,) or (..., 2).
    v2 : array-like
        Second Bloch vector, shape (2,) or (..., 2).
        
    Returns
    -------
    d : ndarray
        Bures distance(s) in [0, √2].
    """
    F = fidelity(v1, v2)
    return np.sqrt(2 * (1 - np.sqrt(F)))


def bures_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute Bures angle (geodesic angle between density matrices).
    
    θ_B(ρ₁, ρ₂) = arccos(√F(ρ₁, ρ₂))
    
    This is the natural geodesic distance on the space of density matrices
    with the Bures-Helstrom metric. For pure states, it equals the
    Fubini-Study distance.
    
    Note: The term "Fubini-Study distance" is canonically reserved for
    pure states in projective Hilbert space. For mixed states (which
    is what Bloch vectors with ||v|| < 1 represent), the correct term
    is "Bures angle."
    
    Parameters
    ----------
    v1 : array-like
        First Bloch vector, shape (2,) or (..., 2).
    v2 : array-like
        Second Bloch vector, shape (2,) or (..., 2).
        
    Returns
    -------
    d : ndarray
        Bures angle(s) in [0, π/2].
    """
    F = fidelity(v1, v2)
    sqrt_F = np.sqrt(np.clip(F, 0, 1))
    return np.arccos(sqrt_F)


# Alias for backwards compatibility and clarity
def fubini_study_distance(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Alias for bures_angle (Fubini-Study is the pure-state special case)."""
    return bures_angle(v1, v2)


def relative_entropy(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute quantum relative entropy (Kullback-Leibler divergence).
    
    S(ρ₁ || ρ₂) = tr(ρ₁ (log ρ₁ - log ρ₂))
    
    Warning: Undefined when supp(ρ₁) ⊄ supp(ρ₂).
    For pure states, this is +∞ unless ρ₁ = ρ₂.
    
    Parameters
    ----------
    v1 : array-like
        First Bloch vector, shape (2,) or (..., 2).
    v2 : array-like
        Second Bloch vector, shape (2,) or (..., 2).
        
    Returns
    -------
    S : ndarray
        Relative entropy values (may be +∞).
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    
    # Build density matrices
    rho1 = rho_of_v(v1)
    rho2 = rho_of_v(v2)
    
    # For 2×2 matrices, compute explicitly
    # Eigenvalues of ρ₁: λ± = ½(1 ± ||v||)
    r1 = np.linalg.norm(v1, axis=-1)
    r2 = np.linalg.norm(v2, axis=-1)
    
    lam1_plus = 0.5 * (1 + r1)
    lam1_minus = 0.5 * (1 - r1)
    lam2_plus = 0.5 * (1 + r2)
    lam2_minus = 0.5 * (1 - r2)
    
    # Relative entropy is complex for non-commuting matrices
    # Use direct matrix computation for accuracy
    
    result = np.zeros_like(r1)
    
    # Handle scalar and array cases
    if r1.ndim == 0:
        rho1 = rho1.reshape(2, 2)
        rho2 = rho2.reshape(2, 2)
        
        # Check if ρ₂ is full rank
        if r2 >= 1 - 1e-10:
            return np.inf
        
        # Compute log(ρ₂) via eigendecomposition
        eig2, U2 = np.linalg.eigh(rho2)
        with np.errstate(divide='ignore'):
            log_eig2 = np.where(eig2 > 1e-15, np.log(eig2), -np.inf)
        log_rho2 = U2 @ np.diag(log_eig2) @ U2.T
        
        # tr(ρ₁ log ρ₁)
        eig1, U1 = np.linalg.eigh(rho1)
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy_terms = np.where(eig1 > 1e-15, eig1 * np.log(eig1), 0)
        neg_entropy = np.sum(entropy_terms)
        
        # tr(ρ₁ log ρ₂)
        cross_term = np.trace(rho1 @ log_rho2)
        
        result = neg_entropy - cross_term
        if np.isinf(result) or np.isnan(result):
            result = np.inf
    else:
        # Array case - use vectorized approximate formula
        # S(ρ₁||ρ₂) ≈ ... (complex for non-commuting)
        # Fallback to loop for accuracy
        result = np.full_like(r1, np.nan)
        for idx in np.ndindex(r1.shape):
            result[idx] = relative_entropy(v1[idx], v2[idx])
    
    return result


def compare_distances(v1: np.ndarray, v2: np.ndarray) -> dict:
    """Compare all distance measures between two Bloch vectors.
    
    Parameters
    ----------
    v1 : array-like
        First Bloch vector, shape (2,).
    v2 : array-like
        Second Bloch vector, shape (2,).
        
    Returns
    -------
    distances : dict
        Dictionary with all distance values.
    """
    from .geometry import hilbert_distance
    
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    
    return {
        'hilbert': float(hilbert_distance(v1, v2)),
        'trace': float(trace_distance(v1, v2)),
        'bures': float(bures_distance(v1, v2)),
        'bures_angle': float(bures_angle(v1, v2)),
        'fidelity': float(fidelity(v1, v2)),
        'euclidean': float(np.linalg.norm(v1 - v2)),
    }


__all__ = [
    "trace_distance",
    "fidelity",
    "bures_distance",
    "bures_angle",
    "fubini_study_distance",  # alias for bures_angle
    "relative_entropy",
    "compare_distances",
]
