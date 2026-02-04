"""Parameter definitions for the chromatic Bloch disk mapping.

Implements the θ tuple from the LaTeX derivations (v8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


def _validate_positive(name: str, value: float, allow_zero: bool = False) -> None:
    """Raise ValueError if value violates positivity constraint."""
    if allow_zero:
        ok = value >= 0
    else:
        ok = value > 0
    if not ok:
        bound = ">=" if allow_zero else ">"
        raise ValueError(f"{name} must be {bound} 0 (got {value})")


@dataclass
class Theta:
    """Container for the θ parameters (Definition: Parameter set θ).

    Fields follow the notation in the derivations:
    - w_L, w_M: luminance weights
    - gamma: opponent mixing term
    - beta: S opponent weight
    - epsilon: luminance floor for projection stability
    - kappa: radial compression gain
    """

    w_L: float = field(default=1.0)
    w_M: float = field(default=1.0)
    gamma: float = field(default=1.0)
    beta: float = field(default=0.5)
    epsilon: float = field(default=0.01)
    kappa: float = field(default=1.0)

    def __post_init__(self) -> None:
        _validate_positive("w_L", self.w_L)
        _validate_positive("w_M", self.w_M)
        _validate_positive("gamma", self.gamma)
        _validate_positive("beta", self.beta)
        _validate_positive("kappa", self.kappa)
        _validate_positive("epsilon", self.epsilon, allow_zero=True)

    @property
    def Delta(self) -> float:  # noqa: N802 (formula symbol)
        """Δ = w_L*gamma + w_M (Eq. Delta definition)."""
        return self.w_L * self.gamma + self.w_M

    @classmethod
    def default(cls) -> "Theta":
        """Return the canonical default parameters."""
        return cls(1.0, 1.0, 1.0, 0.5, 0.01, 1.0)

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize θ to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_dict(self) -> Dict[str, float]:
        """Return a dictionary representation of θ."""
        return {
            "w_L": self.w_L,
            "w_M": self.w_M,
            "gamma": self.gamma,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "kappa": self.kappa,
        }

    @classmethod
    def from_json(cls, data: str | bytes | bytearray | Dict[str, Any]) -> "Theta":
        """Deserialize θ from a JSON string or dict."""
        if isinstance(data, (str, bytes, bytearray)):
            payload = json.loads(data)
        else:
            payload = data
        return cls(
            w_L=float(payload["w_L"]),
            w_M=float(payload["w_M"]),
            gamma=float(payload["gamma"]),
            beta=float(payload["beta"]),
            epsilon=float(payload["epsilon"]),
            kappa=float(payload["kappa"]),
        )

    @classmethod
    def from_whitepoint(
        cls,
        L_white: float,
        M_white: float,
        S_white: float,
        w_L: float = 1.0,
        w_M: float = 1.0,
        epsilon: float = 0.01,
        kappa: float = 1.0,
    ) -> "Theta":
        """Create θ calibrated so a given whitepoint maps to the achromatic axis.

        Given a chosen neutral/white LMS (e.g., D65 white after XYZ→LMS),
        this sets γ and β such that the whitepoint lies on the achromatic
        locus (O₁ = O₂ = 0), meaning it maps to v = (0, 0).

        The calibration formulas are:
            γ = L_white / M_white
            β = S_white / (L_white + M_white)

        This ensures:
            O₁ = L - γM = L_white - (L_white/M_white)*M_white = 0
            O₂ = S - β(L+M) = S_white - [S_white/(L+M)]*(L+M) = 0

        Parameters
        ----------
        L_white, M_white, S_white : float
            LMS coordinates of the chosen whitepoint (e.g., D65 white).
        w_L, w_M : float
            Luminance weights (default 1.0 each).
        epsilon : float
            Luminance floor (default 0.01).
        kappa : float
            Radial compression gain (default 1.0).

        Returns
        -------
        Theta
            Parameter set calibrated to the whitepoint.

        Examples
        --------
        >>> # D65 white in LMS (using HPE matrix)
        >>> # XYZ_D65 = [0.95047, 1.0, 1.08883]
        >>> # M_HPE @ XYZ_D65 ≈ [0.9999, 1.0000, 1.0888]
        >>> theta = Theta.from_whitepoint(0.9999, 1.0000, 1.0888)
        >>> # Now grayscale maps exactly to origin
        """
        if L_white <= 0 or M_white <= 0 or S_white <= 0:
            raise ValueError("Whitepoint LMS values must be positive")

        gamma = L_white / M_white
        beta = S_white / (L_white + M_white)

        return cls(
            w_L=w_L,
            w_M=w_M,
            gamma=gamma,
            beta=beta,
            epsilon=epsilon,
            kappa=kappa,
        )


def d65_whitepoint_lms_hpe() -> Tuple[float, float, float]:
    """Return the D65 whitepoint in LMS using the HPE matrix.

    XYZ_D65 = [0.95047, 1.0, 1.08883]
    
    Computation:
        M_HPE @ XYZ_D65 = [0.38971*0.95047 + 0.68898*1.0 - 0.07868*1.08883,
                          -0.22981*0.95047 + 1.18340*1.0 + 0.04641*1.08883,
                          1.08883]
                       ≈ [0.9737, 1.0155, 1.0888]

    Returns
    -------
    L, M, S : tuple of float
        LMS coordinates of D65 white: (0.9737, 1.0155, 1.0888).
    """
    import numpy as np
    xyz_d65 = np.array([0.95047, 1.0, 1.08883])
    M_HPE = np.array([
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0.00000, 0.00000, 1.00000],
    ])
    lms = M_HPE @ xyz_d65
    return float(lms[0]), float(lms[1]), float(lms[2])


__all__ = ["Theta", "d65_whitepoint_lms_hpe"]
