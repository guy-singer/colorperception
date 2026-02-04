"""Luminance-conditioned reconstruction (right-inverse) of Φ_θ.

The reconstruction is:
    Φ̃_θ⁻¹(v; Y_target) = A_θ⁻¹(Π⁻¹(T_κ⁻¹(v); Y_target))

References:
    Definition: Luminance-Conditioned Inverse of Π
    Proposition: Inverse of A_θ
    Definition: Luminance-Conditioned Reconstruction
    Proposition: Explicit Positivity Conditions
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .params import Theta
from .compression import decompress_from_disk


def inverse_pi(
    u: np.ndarray,
    Y_target: np.ndarray | float,
    theta: Theta,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert Π with specified target luminance.

    Definition: Luminance-Conditioned Inverse of Π.

    Given u = (u1, u2) and Y_target:
        Y = Y_target
        O1 = u1 * (Y_target + ε)
        O2 = u2 * (Y_target + ε)

    Parameters
    ----------
    u : array-like
        Chromaticity coordinates, shape (..., 2).
    Y_target : array-like or float
        Target luminance, broadcastable to u's leading dims.
    theta : Theta
        Parameter set.

    Returns
    -------
    Y, O1, O2 : tuple of ndarray
        Opponent coordinates.
    """
    u = np.asarray(u, dtype=float)
    Y_target = np.asarray(Y_target, dtype=float)

    u1 = u[..., 0]
    u2 = u[..., 1]

    factor = Y_target + theta.epsilon
    O1 = u1 * factor
    O2 = u2 * factor

    return Y_target, O1, O2


def inverse_opponent(
    Y: np.ndarray,
    O1: np.ndarray,
    O2: np.ndarray,
    theta: Theta,
) -> np.ndarray:
    """Invert the opponent transform A_θ.

    Proposition: Inverse of A_θ.

    A_θ⁻¹ = (1/Δ) * [[γ, w_M, 0], [1, -w_L, 0], [β(1+γ), β(w_M-w_L), Δ]]

    So:
        L = (γ*Y + w_M*O1) / Δ
        M = (Y - w_L*O1) / Δ
        S = (β*(1+γ)*Y + β*(w_M-w_L)*O1 + Δ*O2) / Δ

    Parameters
    ----------
    Y : array-like
        Luminance.
    O1 : array-like
        First opponent coordinate.
    O2 : array-like
        Second opponent coordinate.
    theta : Theta
        Parameter set.

    Returns
    -------
    lms : ndarray
        LMS coordinates, shape (..., 3).
    """
    Y = np.asarray(Y, dtype=float)
    O1 = np.asarray(O1, dtype=float)
    O2 = np.asarray(O2, dtype=float)

    Delta = theta.Delta

    L = (theta.gamma * Y + theta.w_M * O1) / Delta
    M = (Y - theta.w_L * O1) / Delta
    S = (
        theta.beta * (1.0 + theta.gamma) * Y
        + theta.beta * (theta.w_M - theta.w_L) * O1
        + Delta * O2
    ) / Delta

    return np.stack([L, M, S], axis=-1)


def reconstruct_lms(
    v: np.ndarray,
    Y_target: np.ndarray | float,
    theta: Theta,
) -> np.ndarray:
    """Full reconstruction: Φ̃_θ⁻¹(v; Y_target) → LMS.

    Definition: Luminance-Conditioned Reconstruction.

    Steps:
        1. u = T_κ⁻¹(v)
        2. (Y, O1, O2) = Π⁻¹(u; Y_target)
        3. LMS = A_θ⁻¹(Y, O1, O2)

    Parameters
    ----------
    v : array-like
        Bloch disk coordinates, shape (..., 2).
    Y_target : array-like or float
        Target luminance.
    theta : Theta
        Parameter set.

    Returns
    -------
    lms : ndarray
        Reconstructed LMS coordinates, shape (..., 3).
    """
    v = np.asarray(v, dtype=float)

    u = decompress_from_disk(v, theta)
    Y, O1, O2 = inverse_pi(u, Y_target, theta)
    lms = inverse_opponent(Y, O1, O2, theta)

    return lms


def positivity_conditions(
    v: np.ndarray,
    Y_target: np.ndarray | float,
    theta: Theta,
) -> Dict[str, Any]:
    """Check positivity feasibility conditions for reconstruction.

    Proposition: Explicit Positivity Conditions.

    Let u = T_κ⁻¹(v). The conditions for L, M, S > 0 are:
        L > 0 ⟺ Y*(γ + w_M*u1) + w_M*u1*ε > 0
        M > 0 ⟺ Y*(1 - w_L*u1) - w_L*u1*ε > 0
        S > 0 ⟺ Y*A(u) + ε*B(u) > 0

    where:
        A(u) = β(1+γ) + β(w_M-w_L)*u1 + Δ*u2
        B(u) = β(w_M-w_L)*u1 + Δ*u2

    Parameters
    ----------
    v : array-like
        Bloch disk coordinates, shape (..., 2).
    Y_target : array-like or float
        Target luminance.
    theta : Theta
        Parameter set.

    Returns
    -------
    dict
        Contains:
        - 'margin_L', 'margin_M', 'margin_S': the LHS of each inequality
        - 'L_pos', 'M_pos', 'S_pos': boolean arrays
        - 'A_u', 'B_u': the A(u) and B(u) functions
        - 'u': the decompressed chromaticity
    """
    v = np.asarray(v, dtype=float)
    Y = np.asarray(Y_target, dtype=float)

    u = decompress_from_disk(v, theta)
    u1 = u[..., 0]
    u2 = u[..., 1]

    eps = theta.epsilon
    Delta = theta.Delta

    # A(u) = β(1+γ) + β(w_M-w_L)*u1 + Δ*u2 = Δ*(u2 - g(u1))
    A_u = theta.beta * (1.0 + theta.gamma) + theta.beta * (theta.w_M - theta.w_L) * u1 + Delta * u2
    # B(u) = β(w_M-w_L)*u1 + Δ*u2
    B_u = theta.beta * (theta.w_M - theta.w_L) * u1 + Delta * u2

    # Margins (LHS of each inequality)
    margin_L = Y * (theta.gamma + theta.w_M * u1) + theta.w_M * u1 * eps
    margin_M = Y * (1.0 - theta.w_L * u1) - theta.w_L * u1 * eps
    margin_S = Y * A_u + eps * B_u

    return {
        "margin_L": margin_L,
        "margin_M": margin_M,
        "margin_S": margin_S,
        "L_pos": margin_L > 0,
        "M_pos": margin_M > 0,
        "S_pos": margin_S > 0,
        "A_u": A_u,
        "B_u": B_u,
        "u": u,
    }


def minimum_luminance_required(
    v: np.ndarray,
    theta: Theta,
) -> np.ndarray:
    """Compute minimum luminance Y for positive LMS reconstruction.

    Remark: Minimum Luminance for Feasibility.

    The lower bounds from the positivity conditions are:
        - If u1 > 0: Y > w_L*u1*ε / (1 - w_L*u1) for M > 0
        - If u1 < 0: Y > -w_M*u1*ε / (γ + w_M*u1) for L > 0
        - If B(u) < 0: Y > -ε*B(u) / A(u) for S > 0

    Returns the maximum of these lower bounds.

    Parameters
    ----------
    v : array-like
        Bloch disk coordinates, shape (..., 2).
    theta : Theta
        Parameter set.

    Returns
    -------
    Y_min : ndarray
        Minimum required luminance, shape (...).
        inf if feasibility is impossible.
    """
    v = np.asarray(v, dtype=float)
    u = decompress_from_disk(v, theta)
    u1 = u[..., 0]
    u2 = u[..., 1]

    eps = theta.epsilon
    Delta = theta.Delta

    # Initialize with zeros (no constraint)
    Y_min = np.zeros_like(u1)

    # A(u) and B(u)
    A_u = theta.beta * (1.0 + theta.gamma) + theta.beta * (theta.w_M - theta.w_L) * u1 + Delta * u2
    B_u = theta.beta * (theta.w_M - theta.w_L) * u1 + Delta * u2

    # Constraint from M > 0 when u1 > 0
    # Y > w_L*u1*ε / (1 - w_L*u1)
    # Denominator must be > 0, i.e., u1 < 1/w_L
    mask_u1_pos = u1 > 0
    denom_M = 1.0 - theta.w_L * u1
    with np.errstate(divide="ignore", invalid="ignore"):
        bound_M = np.where(
            mask_u1_pos & (denom_M > 0),
            theta.w_L * u1 * eps / denom_M,
            np.where(mask_u1_pos & (denom_M <= 0), np.inf, 0.0),
        )
    Y_min = np.maximum(Y_min, bound_M)

    # Constraint from L > 0 when u1 < 0
    # Y > -w_M*u1*ε / (γ + w_M*u1)
    mask_u1_neg = u1 < 0
    denom_L = theta.gamma + theta.w_M * u1
    with np.errstate(divide="ignore", invalid="ignore"):
        bound_L = np.where(
            mask_u1_neg & (denom_L > 0),
            -theta.w_M * u1 * eps / denom_L,
            np.where(mask_u1_neg & (denom_L <= 0), np.inf, 0.0),
        )
    Y_min = np.maximum(Y_min, bound_L)

    # Constraint from S > 0 when B(u) < 0
    # Y > -ε*B(u) / A(u)
    mask_B_neg = B_u < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        bound_S = np.where(
            mask_B_neg & (A_u > 0),
            -eps * B_u / A_u,
            np.where(mask_B_neg & (A_u <= 0), np.inf, 0.0),
        )
    Y_min = np.maximum(Y_min, bound_S)

    return Y_min


def reconstruct_from_phi_roundtrip(
    lms: np.ndarray,
    theta: Theta,
) -> np.ndarray:
    """Test helper: compute roundtrip error ||LMS - reconstruct(Φ(LMS), Y)||.

    Parameters
    ----------
    lms : array-like
        Original LMS coordinates, shape (..., 3).
    theta : Theta
        Parameter set.

    Returns
    -------
    error : ndarray
        Reconstruction error norms, shape (...).
    """
    from .mapping import phi_theta
    from .opponent import opponent_transform

    lms = np.asarray(lms, dtype=float)

    # Forward map
    v = phi_theta(lms, theta)

    # Original luminance
    Y, _, _ = opponent_transform(lms, theta)

    # Reconstruct
    lms_reconstructed = reconstruct_lms(v, Y, theta)

    # Error
    error = np.linalg.norm(lms_reconstructed - lms, axis=-1)
    return error


__all__ = [
    "inverse_pi",
    "inverse_opponent",
    "reconstruct_lms",
    "positivity_conditions",
    "minimum_luminance_required",
    "reconstruct_from_phi_roundtrip",
]
