# Chromabloch Implementation Report

## 1. Overview

This report documents the implementation of the chromaticity map Φ_θ from LMS cone responses to the chromatic Bloch disk, following the mathematical derivations in "Part I derivations v8.tex".

### 1.1 Mathematical Summary

The chromaticity map is defined as the composition:

```
Φ_θ = T_κ ∘ Π ∘ O : ℝ_{>0}³ → D
```

where D = {v ∈ ℝ² : ||v|| < 1} is the open unit disk.

**Stage 1: Opponent Transform O**
```
O(L, M, S) = [Y, O₁, O₂]ᵀ = A_θ · [L, M, S]ᵀ
```
where:
- Y = w_L·L + w_M·M (luminance)
- O₁ = L - γ·M (red-green opponent)
- O₂ = S - β·(L + M) (yellow-blue opponent)

**Stage 2: Chromaticity Projection Π**
```
Π(Y, O₁, O₂) = (O₁/(Y + ε), O₂/(Y + ε)) = u
```

**Stage 3: Radial Compression T_κ**
```
T_κ(u) = tanh(κ||u||) · u/||u||
```
mapping ℝ² → D.

**Stage 4: Density Matrix Representation**
```
ρ(v) = ½(I₂ + v₁σ₁ + v₂σ₂)
```
where σ₁, σ₂ are the real Pauli matrices.

---

## 2. Implementation Details

### 2.1 Module Organization

| Module | Purpose |
|--------|---------|
| `params.py` | Theta dataclass with validation and serialization |
| `opponent.py` | Opponent transform O and matrix A_θ |
| `compression.py` | Radial compression T_κ and inverse |
| `mapping.py` | Full map Φ_θ and chromaticity projection Π |
| `density.py` | Density matrix ρ(v), entropy, saturation, hue |
| `reconstruction.py` | Luminance-conditioned inverse and positivity checks |
| `geometry.py` | Hilbert distance and Klein gyroaddition |
| `mathutils.py` | Attainable region boundary and validation |
| `demo.py` | Artifact generation for validation |

### 2.2 Numerical Stability

**Compression near origin (T_κ for ||u|| → 0):**
- Threshold: r₀ = 10⁻⁸
- Series expansion: tanh(κr)/r ≈ κ - κ³r²/3
- Implementation: `compression.py`, lines 22-31

**Decompression near origin (T_κ⁻¹ for ||v|| → 0):**
- Series expansion: arctanh(r)/r ≈ 1 + r²/3
- Implementation: `compression.py`, lines 50-58

**Decompression near boundary (||v|| → 1):**
- Clamp: r_safe = min(||v||, 1 - 10⁻¹²)
- Implementation: `compression.py`, line 48

**Boundary clamping after compression:**
- Ensure ||v|| < 1 by scaling: v *= (1 - 10⁻¹²)/||v|| if needed
- Implementation: `compression.py`, lines 36-40

**Hilbert distance numerical stability:**
- Clamp squared norms: ||p||², ||q||² ≤ 1 - 10⁻¹⁵
- Clamp arcosh argument: max(arg, 1.0)
- Implementation: `geometry.py`, lines 46-58

### 2.3 Clamping Policies

| Operation | Threshold | Policy |
|-----------|-----------|--------|
| Compression scale | r < 10⁻⁸ | Use Taylor series |
| Decompression scale | r < 10⁻⁸ | Use Taylor series |
| Disk boundary | ||v|| ≥ 1 | Scale to (1-10⁻¹²)·v̂ |
| arctanh input | r ≥ 1 | Clamp to 1-10⁻¹² |
| arcosh argument | arg < 1 | Clamp to 1.0 |

---

## 3. Validation

### 3.1 Test Coverage

| Test Module | Tests | Key Validations |
|-------------|-------|-----------------|
| `test_opponent.py` | 8 | Matrix form, achromatic point, determinant |
| `test_compression.py` | 12 | Origin mapping, disk boundedness, round-trip |
| `test_density.py` | 18 | ρ properties, trace=1, det formula, round-trip |
| `test_attainable_region.py` | 10 | Boundary function, LMS→region, sufficiency |
| `test_mapping_roundtrip.py` | 14 | Full round-trip, achromatic locus, positivity |
| `test_geometry.py` | 18 | Hilbert distance, triangle inequality, gyroaddition |

### 3.2 Critical Identity Tests

**Density matrix round-trip:**
```
v = bloch_from_rho(rho_of_v(v))
```
Tested in `test_density.py::TestBlochFromRho::test_roundtrip`

**Factor-of-2 correctness:**
```
v₂ = 2b (NOT v₂ = b)
```
Tested in `test_density.py::TestBlochFromRho::test_factor_of_two_correct`

**Compression round-trip:**
```
u = T_κ⁻¹(T_κ(u))
```
Tested in `test_compression.py::TestRoundTrip`

**Full mapping round-trip:**
```
LMS = Φ̃_θ⁻¹(Φ_θ(LMS); Y(LMS))
```
Tested in `test_mapping_roundtrip.py::TestReconstructLms`

**Attainable region membership:**
```
For all (L,M,S) ∈ ℝ_{>0}³: u⁽⁰⁾(L,M,S) ∈ attainable region
```
Tested in `test_attainable_region.py::TestRandomLMSInRegion`

**Hilbert distance via gyroaddition:**
```
d_H(u, v) = arctanh(||(-u) ⊕ v||)
```
Tested in `test_geometry.py::TestKleinGyroadd::test_distance_via_gyroaddition`

### 3.3 Independent Validation

To strengthen confidence beyond round-trip tests (which can be "self-referential"), we added independent validation:

**Hilbert Distance: Klein vs Cross-Ratio**

Two completely different computational methods for the same quantity:

1. **Klein formula**: `d_H = arcosh((1 - ⟨p,q⟩) / √((1-||p||²)(1-||q||²)))`
2. **Cross-ratio formula**: `d_H = (1/2)|log((|a⁺-p||a⁻-q|) / (|a⁺-q||a⁻-p|))|`

where a⁺, a⁻ are the boundary intersection points of the line through p and q.

Test results:
- 100 random point pairs: Maximum deviation < 10⁻⁸
- Points near origin: Maximum deviation < 10⁻⁷
- Points near boundary: Maximum deviation < 10⁻⁶
- Collinear with origin: Matches `arctanh` formula to 10⁻¹⁰

This provides strong evidence that the geometry implementation is correct.

### 3.4 Zero-Input Handling

**Policy**: When ε > 0 (default), exact zeros are safe inputs.

| Input | ε = 0 | ε > 0 |
|-------|-------|-------|
| Black (0,0,0) | NaN/Inf (Y=0) | (0,0) ✓ |
| Single cone [L,0,0] | May work | Works ✓ |
| Near-zero [1e-10,...] | Unstable | Stable ✓ |

**Implementation**: Demo pipelines use `np.maximum(lms, 0.0)` (not `1e-10`) when ε > 0.

### 3.5 Tolerances

| Test Category | Relative Tolerance | Absolute Tolerance |
|---------------|-------------------|-------------------|
| Round-trip LMS | 10⁻⁹ | — |
| Round-trip v | 10⁻¹⁰ | — |
| Achromatic mapping | — | 10⁻¹⁰ |
| Distance symmetry | 10⁻¹⁰ | — |
| Triangle inequality | — | 10⁻¹⁰ |

---

## 4. Generated Artifacts

Running `python -m chromabloch.demo` generates:

### 4.1 Plots

| File | Description |
|------|-------------|
| `bloch_scatter.png` | Random LMS samples mapped to Bloch disk, colored by hue |
| `u_region.png` | Attainable chromaticity region with boundary line |
| `saturation_hue_wheel.png` | Polar visualization of entropy and saturation |

### 4.2 Data Files

| File | Contents |
|------|----------|
| `theta.json` | Parameter values used for generation |
| `run_info.json` | Timestamp, Python/NumPy versions, seed, git commit |
| `arrays.npz` | NumPy arrays: lms, v, r, hue, entropy, saturation |

### 4.3 Storage Convention

All outputs are stored in:
```
results/YYYYMMDD_HHMMSS/
├── plots/
│   ├── bloch_scatter.png
│   ├── u_region.png
│   └── saturation_hue_wheel.png
├── theta.json
├── run_info.json
└── arrays.npz
```

Previous runs are never overwritten.

---

## 5. Known Limitations

### 5.1 LMS Conversion is External

This package assumes LMS cone responses as input. The XYZ→LMS conversion is **explicitly external** to Part I of the theory. Different matrix choices (HPE, CAT02, Stockman-Sharpe) will produce different results. The `examples/` directory includes HPE-based demos, but this is a placeholder, not a validated choice.

### 5.2 Float64 tanh Saturation

**Critical numerical limitation**: For float64, `tanh(x) ≈ 1.0` (indistinguishable) when x > ~18.4.

This means:
- When κ||u|| > 18, the compression loses information
- Reconstruction via `arctanh(||v||)` cannot recover the original ||u||
- Hyperbolic distances become capped at ~18.4

**API Contract**: The right-inverse Φ̃θ⁻¹ ∘ Φθ ≈ id holds **ONLY** when κ||u|| < 15 (conservatively).

**Mitigation**:
- Use `compression_saturation_diagnostics(u, theta)` to detect saturation
- Use `suggest_kappa_for_max_u_norm(max_norm)` to choose κ for your data
- Run `examples/srgb_grid_analysis.py` to calibrate κ for the sRGB gamut

### 5.3 Open Disk Representation

The LaTeX defines the Bloch disk as **open**: D = {v : ||v|| < 1}. Numerically, we enforce:
- After compression: scale v to (1 - 10⁻¹²)·v̂ if ||v|| ≥ 1
- All functions use `DISK_EPS = 1e-12` for boundary handling

When printing or displaying ||v||, values like `1.0000` may actually be `0.999999999999` internally.

### 5.4 Parameters Not Calibrated

The default θ values are **mathematical placeholders**, not psychophysically validated:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| w_L, w_M | 1.0, 1.0 | Equal luminance weighting |
| γ | 1.0 | L/M balance for achromatic |
| β | 0.5 | S/(L+M) balance for achromatic |
| ε | 0.01 | Luminance floor |
| κ | 1.0 | Compression gain |

**Whitepoint calibration**: Use `Theta.from_whitepoint(L, M, S)` to set γ, β so that a chosen neutral maps to the origin.

### 5.5 Reconstruction Limitations

1. **Positivity not guaranteed**: `reconstruct_lms(v, Y)` may produce negative L, M, or S for chromaticities outside the attainable region at the given luminance. Use `positivity_conditions(v, Y, theta)` before reconstruction.

2. **Saturation regime**: Reconstruction is numerically unreliable when the original κ||u|| exceeded ~18. No warning is raised; the result will simply be incorrect.

### 5.6 Perceptual Validity Not Established

This implementation is a **faithful realization of Part I mathematics**. It does **NOT** establish:
- Hue angles match human hue perception
- Hilbert distance correlates with discrimination thresholds (JNDs)
- Entropy-based saturation correlates with perceived saturation
- Default parameters are meaningful for real observers

Perceptual validation requires Part II calibration work.

### 5.7 Single Observer Model

No inter-observer variability or chromatic adaptation states are modeled. The mapping is deterministic for a fixed θ.

---

## 6. Next Steps for Part II

1. **Parameter calibration**: Fit θ to MacAdam ellipse data or modern color-difference datasets.

2. **LMS conversion selection**: Choose and validate an XYZ→LMS matrix (e.g., CAT02, Hunt-Pointer-Estevez).

3. **Observer-specific domains**: Estimate convex subsets Ω ⊂ D with Hilbert/Finsler geometry matched to discrimination ellipses.

4. **Context effects**: Model background-dependent transformations using structured state maps (channels).

5. **Psychophysical evaluation**: Compare predicted distances against human discrimination thresholds.

---

## 7. References

1. Part I derivations v8.tex (this repository)
2. Berthier, M. (2020). Geometry of color perception. Part 2: perceived colors from real quantum states and Hering's rebit. *J. Math. Neurosci.*, 10:14.
3. Provenzi, E. (2020). Geometry of color perception. Part 1: structures and metrics of a homogeneous color space. *J. Math. Neurosci.*, 10:7.
4. Prencipe, N. (2022). *Théorie et applications d'une nouvelle formulation de l'espace des couleurs perçues*. PhD thesis, Univ. La Rochelle.
