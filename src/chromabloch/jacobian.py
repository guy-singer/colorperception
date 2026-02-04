"""Jacobian computation for the chromaticity map Φ_θ.

Provides both finite-difference reference and analytic Jacobian for sensitivity
analysis and Phase II calibration readiness.

The Jacobian ∂Φ_θ/∂(L,M,S) is a 2×3 matrix relating infinitesimal LMS changes
to Bloch disk position changes.
"""

from __future__ import annotations

import numpy as np

from .params import Theta
from .mapping import phi_theta


def jacobian_phi_finite_diff(
    lms: np.ndarray,
    theta: Theta,
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute Jacobian of Φ_θ using central finite differences.
    
    This is a reference implementation for testing analytic Jacobians.
    
    Parameters
    ----------
    lms : array-like
        LMS cone responses, shape (3,) for single point or (..., 3) for batch.
    theta : Theta
        Parameter set.
    eps : float
        Finite difference step size.
        
    Returns
    -------
    J : ndarray
        Jacobian matrix, shape (..., 2, 3).
        J[..., i, j] = ∂v_i/∂lms_j
    """
    lms = np.asarray(lms, dtype=float)
    original_shape = lms.shape
    single_point = (lms.ndim == 1)
    
    if single_point:
        lms = lms.reshape(1, 3)
    
    n_points = lms.shape[0] if lms.ndim == 2 else np.prod(lms.shape[:-1])
    lms_flat = lms.reshape(-1, 3)
    
    J = np.zeros((n_points, 2, 3), dtype=float)
    
    for j in range(3):  # L, M, S
        # Forward step
        lms_plus = lms_flat.copy()
        lms_plus[:, j] += eps
        v_plus = phi_theta(lms_plus, theta)
        
        # Backward step
        lms_minus = lms_flat.copy()
        lms_minus[:, j] -= eps
        v_minus = phi_theta(lms_minus, theta)
        
        # Central difference
        J[:, :, j] = (v_plus - v_minus) / (2 * eps)
    
    if single_point:
        return J[0]
    else:
        return J.reshape(original_shape[:-1] + (2, 3))


def jacobian_phi_analytic(
    lms: np.ndarray,
    theta: Theta,
) -> np.ndarray:
    """Compute analytic Jacobian of Φ_θ.
    
    Uses chain rule through the three stages:
        ∂v/∂lms = (∂v/∂u)(∂u/∂(Y,O1,O2))(∂(Y,O1,O2)/∂lms)
    
    Parameters
    ----------
    lms : array-like
        LMS cone responses, shape (3,) or (..., 3).
    theta : Theta
        Parameter set.
        
    Returns
    -------
    J : ndarray
        Jacobian matrix, shape (..., 2, 3).
    """
    lms = np.asarray(lms, dtype=float)
    single_point = (lms.ndim == 1)
    
    if single_point:
        lms = lms.reshape(1, 3)
    
    L, M, S = lms[..., 0], lms[..., 1], lms[..., 2]
    
    # Stage 1: ∂(Y, O1, O2)/∂(L, M, S) = A_θ
    # This is the constant opponent matrix
    dY_dL = theta.w_L
    dY_dM = theta.w_M
    dY_dS = 0.0
    
    dO1_dL = 1.0
    dO1_dM = -theta.gamma
    dO1_dS = 0.0
    
    dO2_dL = -theta.beta
    dO2_dM = -theta.beta
    dO2_dS = 1.0
    
    # Compute intermediate values
    Y = theta.w_L * L + theta.w_M * M
    O1 = L - theta.gamma * M
    O2 = S - theta.beta * (L + M)
    
    denom = Y + theta.epsilon
    denom_sq = denom ** 2
    
    # Stage 2: ∂u/∂(Y, O1, O2)
    # u1 = O1 / (Y + ε), u2 = O2 / (Y + ε)
    # ∂u1/∂Y = -O1/(Y+ε)², ∂u1/∂O1 = 1/(Y+ε), ∂u1/∂O2 = 0
    # ∂u2/∂Y = -O2/(Y+ε)², ∂u2/∂O1 = 0, ∂u2/∂O2 = 1/(Y+ε)
    
    u1 = O1 / denom
    u2 = O2 / denom
    
    du1_dY = -O1 / denom_sq
    du1_dO1 = 1.0 / denom
    du1_dO2 = np.zeros_like(L)
    
    du2_dY = -O2 / denom_sq
    du2_dO1 = np.zeros_like(L)
    du2_dO2 = 1.0 / denom
    
    # Chain rule: ∂u/∂lms
    du1_dL = du1_dY * dY_dL + du1_dO1 * dO1_dL + du1_dO2 * dO2_dL
    du1_dM = du1_dY * dY_dM + du1_dO1 * dO1_dM + du1_dO2 * dO2_dM
    du1_dS = du1_dY * dY_dS + du1_dO1 * dO1_dS + du1_dO2 * dO2_dS
    
    du2_dL = du2_dY * dY_dL + du2_dO1 * dO1_dL + du2_dO2 * dO2_dL
    du2_dM = du2_dY * dY_dM + du2_dO1 * dO1_dM + du2_dO2 * dO2_dM
    du2_dS = du2_dY * dY_dS + du2_dO1 * dO1_dS + du2_dO2 * dO2_dS
    
    # Stage 3: ∂v/∂u (compression)
    # v = tanh(κ||u||) * u/||u||
    # For v = f(||u||) * u/||u|| with f(r) = tanh(κr):
    # ∂v/∂u has special structure involving radial and tangential components
    
    u_norm_sq = u1**2 + u2**2
    u_norm = np.sqrt(u_norm_sq)
    
    kappa = theta.kappa
    kappa_r = kappa * u_norm
    
    # Handle small u_norm (Taylor expansion)
    small = u_norm < 1e-8
    
    tanh_kr = np.tanh(kappa_r)
    sech2_kr = 1.0 / np.cosh(kappa_r)**2
    
    # For v = tanh(κr) * û where r = ||u||, û = u/r:
    # ∂v_i/∂u_j = (tanh(κr)/r) * δ_{ij} + (κ*sech²(κr) - tanh(κr)/r) * (u_i*u_j/r²)
    #
    # Let α = tanh(κr)/r, β = κ*sech²(κr) - tanh(κr)/r
    # Then ∂v/∂u = α*I + β*(û⊗û)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.where(small, kappa, tanh_kr / u_norm)
        beta = np.where(small, 0.0, kappa * sech2_kr - tanh_kr / u_norm)
    
    # ∂v/∂u is a 2×2 matrix at each point
    # ∂v1/∂u1 = α + β*(u1/r)², ∂v1/∂u2 = β*(u1*u2/r²)
    # ∂v2/∂u1 = β*(u1*u2/r²), ∂v2/∂u2 = α + β*(u2/r)²
    
    with np.errstate(divide='ignore', invalid='ignore'):
        u1_hat_sq = np.where(small, 0.5, (u1**2) / u_norm_sq)
        u2_hat_sq = np.where(small, 0.5, (u2**2) / u_norm_sq)
        u1u2_hat = np.where(small, 0.0, (u1 * u2) / u_norm_sq)
    
    dv1_du1 = alpha + beta * u1_hat_sq
    dv1_du2 = beta * u1u2_hat
    dv2_du1 = beta * u1u2_hat
    dv2_du2 = alpha + beta * u2_hat_sq
    
    # Final chain rule: ∂v/∂lms = (∂v/∂u)(∂u/∂lms)
    dv1_dL = dv1_du1 * du1_dL + dv1_du2 * du2_dL
    dv1_dM = dv1_du1 * du1_dM + dv1_du2 * du2_dM
    dv1_dS = dv1_du1 * du1_dS + dv1_du2 * du2_dS
    
    dv2_dL = dv2_du1 * du1_dL + dv2_du2 * du2_dL
    dv2_dM = dv2_du1 * du1_dM + dv2_du2 * du2_dM
    dv2_dS = dv2_du1 * du1_dS + dv2_du2 * du2_dS
    
    # Assemble Jacobian
    J = np.stack([
        np.stack([dv1_dL, dv1_dM, dv1_dS], axis=-1),
        np.stack([dv2_dL, dv2_dM, dv2_dS], axis=-1),
    ], axis=-2)
    
    if single_point:
        return J[0]
    
    return J


def jacobian_norm(
    lms: np.ndarray,
    theta: Theta,
    method: str = "analytic",
) -> np.ndarray:
    """Compute Frobenius norm of the Jacobian (sensitivity measure).
    
    ||J||_F = sqrt(sum(J_ij²))
    
    This measures how sensitive v is to changes in LMS at each point.
    
    Parameters
    ----------
    lms : array-like
        LMS cone responses, shape (..., 3).
    theta : Theta
        Parameter set.
    method : str
        'analytic' or 'finite_diff'.
        
    Returns
    -------
    norm : ndarray
        Jacobian Frobenius norms, shape (...).
    """
    if method == "analytic":
        J = jacobian_phi_analytic(lms, theta)
    else:
        J = jacobian_phi_finite_diff(lms, theta)
    
    return np.linalg.norm(J, axis=(-2, -1))


def jacobian_condition_number(
    lms: np.ndarray,
    theta: Theta,
    method: str = "analytic",
) -> np.ndarray:
    """Compute condition number of the Jacobian.
    
    σ_max / σ_min where σ are singular values.
    
    High condition number indicates ill-conditioning (sensitive to noise).
    
    Parameters
    ----------
    lms : array-like
        LMS cone responses, shape (..., 3).
    theta : Theta
        Parameter set.
    method : str
        'analytic' or 'finite_diff'.
        
    Returns
    -------
    cond : ndarray
        Condition numbers, shape (...).
    """
    if method == "analytic":
        J = jacobian_phi_analytic(lms, theta)
    else:
        J = jacobian_phi_finite_diff(lms, theta)
    
    # J is (..., 2, 3), compute singular values for each 2×3 matrix
    # SVD returns singular values in descending order
    s = np.linalg.svd(J, compute_uv=False)  # shape (..., 2)
    
    # Condition number = max(s) / min(s)
    # Handle zero singular values
    s_max = s[..., 0]
    s_min = np.maximum(s[..., -1], 1e-15)
    
    return s_max / s_min


def jacobian_phi_complex_step(
    lms: np.ndarray,
    theta: Theta,
    h: float = 1e-30,
) -> np.ndarray:
    """Compute Jacobian using complex-step differentiation.
    
    Complex-step derivative: f'(x) ≈ Im(f(x + ih)) / h
    
    This achieves near-machine-precision accuracy for analytic functions
    without subtraction cancellation. Use for validating the analytic Jacobian.
    
    IMPORTANT: This implementation uses holomorphic operations only:
    - sqrt(z²) instead of |z| (which involves conjugation)
    - No np.linalg.norm (uses |z|), no np.abs on complex
    - Branching is done on real parts only
    
    Note: Only works for strictly positive LMS (no clamping/abs operations).
    
    Parameters
    ----------
    lms : array-like
        LMS cone responses, shape (3,) for single point.
    theta : Theta
        Parameter set. Must have epsilon > 0 or lms strictly positive.
    h : float
        Step size (default 1e-30, can be much smaller than finite-diff).
        
    Returns
    -------
    J : ndarray
        Jacobian matrix, shape (2, 3).
    """
    lms = np.asarray(lms, dtype=float)
    if lms.ndim != 1 or lms.shape[0] != 3:
        raise ValueError("Complex-step Jacobian only supports single point (shape (3,))")
    
    J = np.zeros((2, 3), dtype=float)
    
    for j in range(3):
        # Create complex perturbation
        lms_c = lms.astype(complex)
        lms_c[j] += 1j * h
        
        # Compute v using complex arithmetic (holomorphic only!)
        # Opponent transform
        L, M, S = lms_c
        Y = theta.w_L * L + theta.w_M * M
        O1 = L - theta.gamma * M
        O2 = S - theta.beta * (L + M)
        
        # Chromaticity
        denom = Y + theta.epsilon
        u1 = O1 / denom
        u2 = O2 / denom
        
        # Compression (complex-valued, holomorphic)
        # CRITICAL: Use algebraic norm sqrt(u1² + u2²), NOT np.linalg.norm/np.abs
        # np.linalg.norm uses |z| = sqrt(z·conj(z)) which breaks holomorphicity
        u_norm_sq = u1**2 + u2**2
        u_norm = np.sqrt(u_norm_sq)  # Holomorphic: sqrt(u1² + u2²)
        kappa_r = theta.kappa * u_norm
        
        # Complex tanh (holomorphic)
        tanh_kr = np.tanh(kappa_r)
        
        # Scale factor
        # Branch on REAL part only to preserve holomorphicity
        if u_norm.real < 1e-10:
            scale = theta.kappa
        else:
            scale = tanh_kr / u_norm
        
        v1 = scale * u1
        v2 = scale * u2
        
        # Extract imaginary parts
        J[0, j] = v1.imag / h
        J[1, j] = v2.imag / h
    
    return J


def verify_scale_invariance_jacobian(
    lms: np.ndarray,
    theta: Theta,
    method: str = "analytic",
) -> float:
    """Verify that the Jacobian satisfies scale invariance at ε=0.
    
    For ε=0, Φ_θ(t·x) = Φ_θ(x) for all t > 0.
    Differentiating: J(x) · x = 0 (the radial direction is in the null space).
    
    Parameters
    ----------
    lms : array-like
        LMS cone responses, shape (3,).
    theta : Theta
        Parameter set. Should have epsilon=0 for exact invariance.
    method : str
        'analytic', 'finite_diff', or 'complex_step'.
        
    Returns
    -------
    norm : float
        ||J(x) · x||, should be near 0 for ε=0.
    """
    lms = np.asarray(lms, dtype=float).flatten()
    
    if method == "analytic":
        J = jacobian_phi_analytic(lms, theta)
    elif method == "complex_step":
        J = jacobian_phi_complex_step(lms, theta)
    else:
        J = jacobian_phi_finite_diff(lms, theta)
    
    # J is 2×3, lms is 3×1
    # J @ lms should be near zero for scale-invariant map
    radial_derivative = J @ lms
    
    return float(np.linalg.norm(radial_derivative))


__all__ = [
    "jacobian_phi_finite_diff",
    "jacobian_phi_analytic",
    "jacobian_phi_complex_step",
    "jacobian_norm",
    "jacobian_condition_number",
    "verify_scale_invariance_jacobian",
]
