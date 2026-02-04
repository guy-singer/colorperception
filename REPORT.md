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
| `params.py` | Theta dataclass with validation, serialization, and whitepoint calibration |
| `opponent.py` | Opponent transform O and matrix A_θ |
| `compression.py` | Radial compression T_κ, inverse, and saturation diagnostics |
| `mapping.py` | Full map Φ_θ, chromaticity projection Π, strict/image mode handling |
| `density.py` | Density matrix ρ(v), entropy, saturation, hue |
| `reconstruction.py` | Luminance-conditioned inverse and positivity checks |
| `geometry.py` | Hilbert distance (Klein + cross-ratio), Klein gyroaddition |
| `mathutils.py` | Attainable region boundary, sampling, and validation |
| `jacobian.py` | Analytic and finite-difference Jacobian, sensitivity metrics |
| `demo.py` | Artifact generation for validation |

### 2.2 Numerical Stability

**Compression near origin (T_κ for ||u|| → 0):**
- Threshold: r₀ = 10⁻⁸
- Series expansion: tanh(κr)/r ≈ κ - κ³r²/3

**Decompression near origin (T_κ⁻¹ for ||v|| → 0):**
- Series expansion: arctanh(r)/r ≈ 1 + r²/3

**Decompression near boundary (||v|| → 1):**
- Clamp: r_safe = min(||v||, 1 - 10⁻¹²)

**Boundary clamping after compression:**
- Ensure ||v|| < 1 by scaling: v *= (1 - 10⁻¹²)/||v|| if needed

**Hilbert distance numerical stability:**
- Clamp squared norms: ||p||², ||q||² ≤ 1 - 10⁻¹⁵
- Clamp arcosh argument: max(arg, 1.0)

### 2.3 Clamping Policies

| Operation | Threshold | Policy |
|-----------|-----------|--------|
| Compression scale | r < 10⁻⁸ | Use Taylor series |
| Decompression scale | r < 10⁻⁸ | Use Taylor series |
| Disk boundary | ||v|| ≥ 1 | Scale to (1-10⁻¹²)·v̂ |
| arctanh input | r ≥ 1 | Clamp to 1-10⁻¹² |
| arcosh argument | arg < 1 | Clamp to 1.0 |

---

## 3. Domain and Practical Input Handling

### 3.1 Strict vs Image Mode

The mapping supports two operational modes via the `strict_domain` parameter:

| | ε = 0 | ε > 0 |
|---|---|---|
| **strict_domain=True** | Raises if LMS ≤ 0 | Raises if LMS ≤ 0 |
| **strict_domain=False** | Raises if Y + ε ≤ 0 | Clips LMS < 0 to 0 (safe) |

**Strict mode (mathematical validation):**
- Enforces theoretical domain LMS ∈ ℝ³₊₊
- Raises `DomainViolation` on any non-positive input
- Φ_θ is smooth on this domain

**Image mode (practical processing):**
- Clips negative LMS to 0 with warning
- Counts clipped values in diagnostics
- Piecewise smooth (non-smooth at clipping boundary)
- Still raises if ε = 0 and Y ≤ 0 would cause division by zero

### 3.2 Zero-Input Handling

| Input | ε = 0 | ε > 0 |
|-------|-------|-------|
| Black (0,0,0) | Raises (Y=0) | Maps to (0,0) ✓ |
| Single cone [L,0,0] | May work if Y>0 | Works ✓ |
| Near-zero [1e-10,...] | Numerically unstable | Stable ✓ |

**Recommendation:** Use ε > 0 (default 0.01) for all practical applications.

---

## 4. Validation

### 4.1 Test Suite

Tests are organized by module. Run `pytest tests/ -v` for current counts.

| Test Module | Key Validations |
|-------------|-----------------|
| `test_opponent.py` | Matrix form, achromatic point, determinant |
| `test_compression.py` | Origin mapping, disk boundedness, round-trip |
| `test_density.py` | ρ properties, trace=1, det formula, round-trip |
| `test_attainable_region.py` | Boundary function, LMS→region, sufficiency |
| `test_mapping_roundtrip.py` | Full round-trip, achromatic locus, positivity |
| `test_geometry.py` | Hilbert distance, triangle inequality, gyroaddition |
| `test_saturation_stress.py` | tanh saturation, scaling laws, boundary behavior |
| `test_independent_validation.py` | Cross-ratio vs Klein, zero handling, feasibility |
| `test_jacobian.py` | Analytic vs finite-diff, sensitivity metrics |

### 4.2 Critical Identity Tests

**Compression round-trip:**
```
u = T_κ⁻¹(T_κ(u))  — holds when κ||u|| < 15
```

**Full mapping round-trip:**
```
LMS = Φ̃_θ⁻¹(Φ_θ(LMS); Y(LMS))  — holds in non-saturated regime
```

**Scale invariance (ε = 0):**
```
Φ_θ(t·x) = Φ_θ(x) for all t > 0
```

**Scaling law (ε > 0):**
```
u^(ε)(tx) = t(Y+ε)/(tY+ε) · u^(ε)(x)
```

**Hilbert distance via gyroaddition:**
```
d_H(u, v) = arctanh(||(-u) ⊕ v||)
```

### 4.3 Independent Validation

**Hilbert Distance: Klein vs Cross-Ratio**

Two computationally independent methods:

1. **Klein formula**: `d_H = arcosh((1 - ⟨p,q⟩) / √((1-||p||²)(1-||q||²)))`
2. **Cross-ratio formula**: `d_H = (1/2)|log((|a⁺-p||a⁻-q|) / (|a⁺-q||a⁻-p|))|`

Agreement to < 10⁻⁸ across random point pairs.

**Jacobian: Analytic vs Finite-Difference**

- Analytic Jacobian derived via chain rule through all three stages
- Validated against central finite differences
- Agreement to rtol < 10⁻⁴ (limited by finite-diff precision)

### 4.4 Tolerances

| Test Category | Relative Tolerance | Absolute Tolerance |
|---------------|-------------------|-------------------|
| Round-trip LMS | 10⁻⁸ | — |
| Round-trip v | 10⁻¹⁰ | — |
| Achromatic mapping | — | 10⁻¹⁰ |
| Jacobian analytic vs FD | 10⁻⁴ | 10⁻⁸ |

---

## 5. Numerical Contracts

This section explicitly separates mathematical guarantees from numerical behavior.

### Contract A: Mathematical Guarantees (Ideal Map)

**Compression T_κ:**
- T_κ: ℝ² → D is a diffeomorphism (bijective, smooth, smooth inverse)
- For all u ∈ ℝ², ||T_κ(u)|| < 1 (strict inequality)
- T_κ⁻¹ exists and is smooth on D

**Full map Φ_θ:**
- Smooth on ℝ³₊₊ (strictly positive LMS)
- Scale-invariant when ε = 0

**Reconstruction:**
- Φ̃_θ⁻¹(v; Y) is well-defined for v ∈ D and Y > 0
- Φ̃_θ⁻¹(Φ_θ(x); Y(x)) = x exactly (right-inverse identity)

### Contract B: Float64 Numerical Behavior (Implementation)

**Compression:**
- `tanh(x) = 1.0` (indistinguishable in float64) when x > ~18.4
- Warning threshold: κ||u|| > 15
- Saturation threshold: κ||u|| > 18
- **Implementation includes boundary clamping** — not a true diffeomorphism at saturation

**Reconstruction reliability:**
- Roundtrip error < 10⁻⁸ when κ||u|| < 15 (empirically validated)
- Beyond threshold: arctanh(tanh(κ||u||)) ≠ κ||u||, reconstruction unreliable

**Boundary clamping:**
- Implementation clamps ||v|| to 1 - 10⁻¹² if tanh produces exactly 1.0
- Reported ||v|| = 1.0000 may actually be 0.999999999999

### Contract C: Domain Extension (Strict vs Image Mode)

**Strict mode (`strict_domain=True`):**
- Raises `DomainViolation` if any LMS ≤ 0
- Guarantees smoothness claims apply
- Use for mathematical validation

**Image mode (`strict_domain=False`, default):**
- Clips negative LMS to 0 with warning (counted in diagnostics)
- Raises if ε = 0 and Y + ε ≤ 0 (division by zero unavoidable)
- Safe when ε > 0: black pixels map to origin
- **Piecewise smooth only** (non-smooth at clipping boundary)

### Contract D: API Safety Invariants

```python
# Safe usage pattern:
v, diag = phi_theta_with_diagnostics(lms, theta)
if diag.is_safe():
    # is_safe() checks:
    #   - n_negative_clipped == 0
    #   - n_saturated == 0
    #   - n_boundary_clamped == 0
    lms_reconstructed = reconstruct_lms(v, Y, theta)
    # Empirically validated: ||lms_reconstructed - lms|| < 1e-8
else:
    # Inspect diag.summary() for details
    # Reconstruction may be unreliable
```

---

## 6. Generated Artifacts

Running `python -m chromabloch.demo` generates:

### 6.1 Plots

| File | Description |
|------|-------------|
| `bloch_scatter.png` | Random LMS samples mapped to Bloch disk, colored by hue |
| `u_region.png` | Attainable chromaticity region with boundary line |
| `saturation_hue_wheel.png` | Polar visualization of entropy and saturation |

### 6.2 Data Files

| File | Contents |
|------|----------|
| `theta.json` | Parameter values used for generation |
| `run_info.json` | Timestamp, Python/NumPy versions, seed, git commit |
| `arrays.npz` | NumPy arrays: lms, v, r, hue, entropy, saturation |

### 6.3 Storage Convention

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

---

## 7. Known Limitations

### 7.1 LMS Conversion is External

This package assumes LMS cone responses as input. The XYZ→LMS conversion is **explicitly external** to Part I of the theory. Different matrix choices (HPE, CAT02, Stockman-Sharpe) will produce different results. The `examples/` directory includes HPE-based demos, but this is a placeholder, not a validated choice.

### 7.2 Float64 tanh Saturation

**Critical numerical limitation**: For float64, `tanh(x) ≈ 1.0` when x > ~18.4.

This means:
- When κ||u|| > 18, the compression loses information
- Reconstruction via `arctanh(||v||)` cannot recover the original ||u||
- Hyperbolic distances become capped at ~18.4

**API Contract**: The right-inverse Φ̃θ⁻¹ ∘ Φθ ≈ id holds **ONLY** when κ||u|| < 15 (conservatively).

**Mitigation**:
- Use `compression_saturation_diagnostics(u, theta)` to detect saturation
- Use `suggest_kappa_for_max_u_norm(max_norm)` to choose κ for your data
- For sRGB: κ ≤ 2.0 safe; for Rec.2020: κ ≤ 1.4 safe

### 7.3 Open Disk vs Implementation Clamping

The LaTeX defines the Bloch disk as **open**: D = {v : ||v|| < 1}.

**Mathematical (Contract A):** ||v|| < 1 always, T_κ is a true diffeomorphism.

**Numerical (Contract B):** Implementation clamps ||v|| to 1 - 10⁻¹² when tanh saturates. This breaks true invertibility at the boundary.

### 7.4 Parameters Not Calibrated

The default θ values are **mathematical placeholders**, not psychophysically validated:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| w_L, w_M | 1.0, 1.0 | Equal luminance weighting |
| γ | 1.0 | L/M balance for achromatic |
| β | 0.5 | S/(L+M) balance for achromatic |
| ε | 0.01 | Luminance floor |
| κ | 1.0 | Compression gain |

**Whitepoint calibration**: Use `Theta.from_whitepoint(L, M, S)` to set γ, β so that a chosen neutral maps to the origin.

### 7.5 Reconstruction Limitations

1. **Positivity not guaranteed**: `reconstruct_lms(v, Y)` may produce negative L, M, or S for chromaticities outside the attainable region at the given luminance.

2. **Saturation regime**: Reconstruction is numerically unreliable when the original κ||u|| exceeded ~18.

### 7.6 Perceptual Validity Not Established

This implementation is a **faithful realization of Part I mathematics**. It does **NOT** establish:
- Hue angles match human hue perception
- Hilbert distance correlates with discrimination thresholds (JNDs)
- Entropy-based saturation correlates with perceived saturation
- Default parameters are meaningful for real observers

Perceptual validation requires Part II calibration work.

### 7.7 Single Observer Model

No inter-observer variability or chromatic adaptation states are modeled. The mapping is deterministic for a fixed θ.

---

## 8. Next Steps for Part II

1. **Parameter calibration**: Fit θ to MacAdam ellipse data or modern color-difference datasets.

2. **LMS conversion selection**: Choose and validate an XYZ→LMS matrix (e.g., CAT02, Hunt-Pointer-Estevez).

3. **Observer-specific domains**: Estimate convex subsets Ω ⊂ D with Hilbert/Finsler geometry matched to discrimination ellipses.

4. **Context effects**: Model background-dependent transformations using structured state maps (channels).

5. **Psychophysical evaluation**: Compare predicted distances against human discrimination thresholds.

6. **Induced metric analysis**: Use Jacobian to compute pullback metric on LMS space.

---

## 9. References

1. Part I derivations v8.tex (this repository)
2. Berthier, M. (2020). Geometry of color perception. Part 2: perceived colors from real quantum states and Hering's rebit. *J. Math. Neurosci.*, 10:14.
3. Provenzi, E. (2020). Geometry of color perception. Part 1: structures and metrics of a homogeneous color space. *J. Math. Neurosci.*, 10:7.
4. Prencipe, N. (2022). *Théorie et applications d'une nouvelle formulation de l'espace des couleurs perçues*. PhD thesis, Univ. La Rochelle.
