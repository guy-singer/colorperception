"""Chromaticity projection Π and full map Φ_θ (Definition: Chromaticity Map).

The chromaticity map Φ_θ: L → D is the composition:
    Φ_θ = T_κ ∘ Π ∘ O

where:
    O: opponent transform (L,M,S) → (Y, O1, O2)
    Π: chromaticity projection (Y, O1, O2) → u = (u1, u2)
    T_κ: radial compression u → v ∈ D
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .params import Theta
from .opponent import opponent_transform
from .compression import compress_to_disk


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


def phi_theta(lms: np.ndarray, theta: Theta) -> np.ndarray:
    """Compute the full chromaticity map Φ_θ(LMS) → v ∈ D.

    Definition: Chromaticity Map (Φ_θ = T_κ ∘ Π ∘ O).

    Parameters
    ----------
    lms : array-like
        LMS cone responses, shape (..., 3).
    theta : Theta
        Parameter set.

    Returns
    -------
    v : ndarray
        Bloch disk coordinates, shape (..., 2), with ||v|| < 1.
    """
    lms = np.asarray(lms, dtype=float)
    Y, O1, O2 = opponent_transform(lms, theta)
    u = chromaticity_projection(Y, O1, O2, theta)
    v = compress_to_disk(u, theta)
    return v


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


__all__ = ["chromaticity_projection", "phi_theta", "phi_theta_components", "PhiComponents"]
