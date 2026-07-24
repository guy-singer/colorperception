# Chromabloch

**LMS → Chromatic Bloch Disk Mapping, Reconstruction, and Geometry Utilities**

This package implements the chromaticity map Φ_θ from LMS cone responses to the chromatic Bloch disk, as described in the mathematical derivations for modeling color perception using rebit (real qubit) state spaces.

## Overview

The chromaticity map is a composition of three stages:

```
Φ_θ = T_κ ∘ Π ∘ O : ℝ_{>0}³ → D
```

Where:
- **O**: Opponent transform — converts LMS to (Y, O₁, O₂) opponent coordinates
- **Π**: Chromaticity projection — normalizes by luminance to get chromaticity u
- **T_κ**: Radial compression — maps unbounded chromaticity to the open unit disk D

The output coordinates (v₁, v₂) ∈ D can be represented as a rebit density matrix ρ(v), enabling quantum-inspired chromatic computations.

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install numpy scipy matplotlib pytest pydantic rich
```

## Quickstart

```python
from chromabloch.params import Theta
from chromabloch.mapping import phi_theta
from chromabloch.reconstruction import reconstruct_lms
from chromabloch.density import rho_of_v, von_neumann_entropy, hue_angle
from chromabloch.geometry import hilbert_distance

# Create parameter set (or use defaults)
theta = Theta.default()

# Map LMS to Bloch disk
lms = [0.8, 1.0, 0.2]
v = phi_theta(lms, theta)
print(f"Bloch coordinates: v = {v}")  # e.g., [0.123, -0.456]

# Compute chromatic attributes
rho = rho_of_v(v)
entropy = von_neumann_entropy(v)
hue = hue_angle(v)
print(f"Entropy: {entropy:.3f}, Hue: {hue:.3f} rad")

# Reconstruct LMS from v with target luminance
Y = theta.w_L * lms[0] + theta.w_M * lms[1]
lms_reconstructed = reconstruct_lms(v, Y, theta)
print(f"Reconstructed: {lms_reconstructed}")

# Compute Hilbert distance between two colors
v1 = phi_theta([0.8, 1.0, 0.2], theta)
v2 = phi_theta([1.2, 0.9, 0.5], theta)
d = hilbert_distance(v1, v2)
print(f"Hilbert distance: {d:.3f}")
```

## Running Tests

```bash
pytest
```

## Running the Demo

```bash
python -m chromabloch.demo
```

This generates:
- `results/<timestamp>/plots/bloch_scatter.png` — LMS samples mapped to Bloch disk
- `results/<timestamp>/plots/u_region.png` — Attainable chromaticity region
- `results/<timestamp>/plots/saturation_hue_wheel.png` — Entropy/saturation visualization
- `results/<timestamp>/theta.json` — Parameter values used
- `results/<timestamp>/run_info.json` — Metadata (seed, versions, etc.)
- `results/<timestamp>/arrays.npz` — Computed arrays for reproducibility

## Package Structure

```
chromabloch/
├── src/chromabloch/
│   ├── params.py         # Theta parameter dataclass
│   ├── opponent.py       # Opponent transform O
│   ├── compression.py    # Radial compression T_κ
│   ├── mapping.py        # Full map Φ_θ and chromaticity projection Π
│   ├── density.py        # Density matrix ρ(v), entropy, saturation, hue
│   ├── reconstruction.py # Inverse reconstruction and positivity checks
│   ├── geometry.py       # Hilbert distance and Klein gyroaddition
│   ├── mathutils.py      # Attainable region helpers
│   └── demo.py           # Artifact generation demo
├── tests/                # Pytest test suite
├── examples/             # Example scripts
└── results/              # Generated artifacts (gitignored except .gitkeep)
```

## Important Note: LMS Conversion is External

As stated in the mathematical derivations (Remark: LMS Conversion is External):

> The transformation from CIE XYZ (or RGB) to LMS uses a 3×3 matrix whose specific entries depend on the chosen cone fundamentals (e.g., Hunt-Pointer-Estevez, CAT02, or physiological cone fundamentals). **This choice is external to the present construction**: we analyze Φ_θ assuming LMS inputs are given.

This package does **not** provide RGB→LMS or XYZ→LMS conversion. Users must supply LMS values directly or use an external library (e.g., `colour-science`) for conversion.

## Parameters

The parameter vector θ = (w_L, w_M, γ, β, ε, κ) controls the mapping:

| Parameter | Description | Default |
|-----------|-------------|---------|
| w_L, w_M  | Luminance weights | 1.0, 1.0 |
| γ (gamma) | L/M opponent balance | 1.0 |
| β (beta)  | S vs (L+M) balance | 0.5 |
| ε (epsilon) | Luminance regularization | 0.01 |
| κ (kappa) | Saturation rate | 1.0 |

## References

- Part I Derivations v8 (LaTeX document in this repository)
- Berthier (2020), "Geometry of color perception. Part 2: perceived colors from real quantum states and Hering's rebit"
- Provenzi (2020), "Geometry of color perception. Part 1: structures and metrics of a homogeneous color space"

## License

MIT License (or as specified by project requirements)
