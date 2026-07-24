"""Opponent transform 𝒪 and related matrices (Definition: Opponent Transform)."""

from __future__ import annotations

import numpy as np

from .params import Theta


def opponent_transform(lms: np.ndarray, theta: Theta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (Y, O1, O2) from LMS.

    Shapes: lms is (..., 3). Returns arrays broadcasted to leading dims.
    """
    arr = np.asarray(lms, dtype=float)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected last dim=3 for lms, got {arr.shape}")
    L, M, S = arr[..., 0], arr[..., 1], arr[..., 2]
    Y = theta.w_L * L + theta.w_M * M
    O1 = L - theta.gamma * M
    O2 = S - theta.beta * (L + M)
    return Y, O1, O2


def A_theta(theta: Theta) -> np.ndarray:
    """Return A_θ matrix (Eq. A_theta definition)."""
    return np.array(
        [
            [theta.w_L, theta.w_M, 0.0],
            [1.0, -theta.gamma, 0.0],
            [-theta.beta, -theta.beta, 1.0],
        ],
        dtype=float,
    )


def det_A_theta(theta: Theta) -> float:
    """Determinant of A_θ: -(w_L*gamma + w_M) = -Δ."""
    return -(theta.w_L * theta.gamma + theta.w_M)


__all__ = ["opponent_transform", "A_theta", "det_A_theta"]
