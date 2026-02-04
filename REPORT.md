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
| `jacobian.py` | Analytic, finite-diff, and complex-step Jacobian; sensitivity metrics |
| `metric.py` | Klein metric tensor, pullback metric, discrimination ellipsoids |
| `quantum_distances.py` | Trace, Bures distance, Bures angle; fidelity; comparison tools |
| `demo.py` | Artifact generation for validation |

### 2.2 Numerical Stability

**Compression near origin (T_κ for ||u|| → 0):**
- Threshold: r₀ = 10⁻⁸
- Series expansion: tanh(κr)/r ≈ κ - κ³r²/3

**Decompression near origin (T_κ⁻¹ for ||v|| → 0):**
- Series expansion: arctanh(r)/r ≈ 1 + r²/3

**Decompression near boundary (||v|| → 1):**
- Clamp: r_safe = min(||v||, R_MAX) where R_MAX = 1 − 10⁻¹⁵

**Boundary clamping after compression:**
- Ensure ||v|| < 1 by scaling: v *= R_MAX/||v|| if needed

**Hilbert distance numerical stability:**
- Clamp squared norms: ||p||², ||q||² ≤ 1 - 10⁻¹⁵
- Clamp arcosh argument: max(arg, 1.0)

### 2.3 Clamping Policies

| Operation | Threshold | Policy |
|-----------|-----------|--------|
| Compression scale | r < 10⁻⁸ | Use Taylor series |
| Decompression scale | r < 10⁻⁸ | Use Taylor series |
| Disk boundary | ||v|| ≥ R_MAX | Scale to R_MAX·v̂ |
| arctanh input | r ≥ R_MAX | Clamp to R_MAX |
| arcosh argument | arg < 1 | Clamp to 1.0 |

**Constants:**
- R_MAX = 1 − 10⁻¹⁵ ≈ 0.999999999999999
- Invertibility cap: atanh(R_MAX) ≈ 17.6

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

### 5.0 Threshold Vocabulary

Four distinct concepts that must not be conflated:

| Concept | Value | Definition |
|---------|-------|------------|
| **Precision regime** | varies | Max κ||u|| for a given error tolerance (empirical, from profiling) |
| **Warning threshold** | 15.0 | Policy threshold for flagging samples (conservative, not a hard limit) |
| **Invertibility cap** | ~17.6 | atanh(R_MAX) — hard limit on recoverable κ||u|| due to disk clamp |
| **tanh → 1.0 threshold** | ~19+ | Where `np.tanh(x)` returns exactly 1.0 (platform-dependent) |

**Precision regimes** (from `compression_roundtrip_profile.json`):

| Tolerance | Max κ||u|| | Regime |
|-----------|------------|--------|
| 10⁻¹² | < 7.2 | Ultra-precise |
| 10⁻¹⁰ | < 10.1 | High-precision |
| 10⁻⁸ | < 11.7 | Standard (default) |
| 10⁻⁶ | < 15.8 | Degraded |
| 10⁻⁴ | < 17.2 | Marginal |

**Key distinction:**
- The **warning threshold** (15) is a *policy* for is_safe() flagging, not a precision boundary
- The **invertibility cap** (~17.6) is the *implementation* hard limit from R_MAX
- The **tanh saturation** (~19+) is where float64 produces tanh(x) = 1.0, but is superseded by the invertibility cap

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

**Measured reconstruction reliability:**

The following table shows empirically measured roundtrip error `|u - T_κ⁻¹(T_κ(u))|/|u|` from sampling a logspace grid:

| κ||u|| max | Relative error | Regime |
|------------|----------------|--------|
| < 7.2      | < 10⁻¹²        | Ultra-precise |
| < 10.1     | < 10⁻¹⁰        | High-precision |
| < 11.7     | < 10⁻⁸         | Standard (default) |
| < 15.8     | < 10⁻⁶         | Degraded |
| < 17.2     | < 10⁻⁴         | Marginal |
| > 17.6     | **undefined**  | Beyond invertibility cap |

**Disk clamp:** R_MAX = 1 − 10⁻¹⁵, so atanh(R_MAX) ≈ 17.6 is the hard invertibility cap.

**Source:** `examples/roundtrip_precision_profile.py` generates `compression_roundtrip_profile.json` with:
- n_points: 1000 (logspace sampling)
- x_range: [0.1, 25]
- disk_clamp.R_MAX and invertibility_cap

**Note:** These are *empirical bounds* on the sampled grid, not mathematical guarantees. The "warning threshold" (15) and "saturation threshold" (18) in diagnostics are conservative flags.

**Boundary clamping:**
- Implementation clamps ||v|| to R_MAX = 1 − 10⁻¹⁵ if tanh produces values ≥ R_MAX
- This induces a hard **invertibility cap** at atanh(R_MAX) ≈ 17.6
- Any ||v|| reported as ~1.0 is actually clamped to R_MAX

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

if diag.is_reconstructable(tol=1e-8):
    # is_reconstructable(tol) uses empirically measured error profile:
    #   - is_safe() == True
    #   - max_kappa_u < max_x_for_reconstruction_tolerance(tol)
    # Expected: reconstruction relative error < tol (empirically validated)
    lms_reconstructed = reconstruct_lms(v, Y, theta)
    
elif diag.is_safe():
    # is_safe() checks:
    #   - n_negative_clipped == 0
    #   - n_saturated == 0
    #   - n_boundary_clamped == 0
    #   - max_kappa_u < 15.0 (warning threshold)
    # Reconstruction likely accurate based on profiling
    lms_reconstructed = reconstruct_lms(v, Y, theta)
    
else:
    # Inspect diag.summary() for details
    # Reconstruction may be unreliable
    print(diag.summary())
```

**Tolerance-aware reconstruction:**
```python
# For precision-critical applications:
if diag.is_reconstructable(tol=1e-10):  # Stricter threshold
    # max_kappa_u < 10.0 required
    ...
    
# For relaxed requirements:
if diag.is_reconstructable(tol=1e-6):   # More permissive
    # max_kappa_u < 15.5 sufficient
    ...
```

**Note:** Thresholds are derived from empirical profiling on a logspace grid, not from analytical bounds.

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

### 6.4 Reproducing Figures

All figures can be regenerated with the master script:

```bash
cd chromabloch
python examples/run_all_figures.py
```

This generates a timestamped manifest with:
- Git commit hash
- Python/NumPy versions
- θ parameters used
- List of generated files

**Individual figure scripts:**

| Figure | Script | Key Parameters |
|--------|--------|----------------|
| `realistic_colors_demo.png` | `demo_realistic_colors.py` | sRGB primaries, ColorChecker patches, HPE matrix |
| `srgb_grid_analysis.png` | `srgb_grid_analysis.py` | 64³ sRGB grid, κ=1.0 |
| `display_p3_analysis.png`, `rec.2020_analysis.png` | `wide_gamut_analysis.py` | 64³ grids per gamut |
| `gamut_boundary_analysis.png`, `kappa_sensitivity.png` | `gamut_boundary_analysis.py` | θ default and D65, κ ∈ {0.5, 1.0, 2.0} |
| `discrimination_ellipses.png`, `distance_comparison.png`, `sensitivity_heatmap.png` | `metric_analysis.py` | D65-calibrated θ |
| `image_demo_*.png` | `image_hue_saturation_demo.py` | Synthetic images |

**Exact reproduction:**
- All scripts use `np.random.default_rng(42)` for determinism
- HPE matrix: Hunt-Pointer-Estevez (see `wide_gamut_analysis.py`)
- Whitepoint: D65 in XYZ = (0.95047, 1.0, 1.08883)

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

**Numerical (Contract B):** Implementation clamps ||v|| to R_MAX = 1 − 10⁻¹⁵. Beyond atanh(R_MAX) ≈ 17.6, reconstruction cannot recover the original κ||u||.

### 7.4 Non-Surjectivity: Φ_θ Does NOT Cover All of D

**Critical**: The attainable region Φ_θ(ℝ³₊₊) is a **proper subset** of D for all finite κ.

The mapping T_κ compresses ℝ² → D via tanh(κ||u||), but the attainable chromaticity region in u-space is itself bounded:
- u₁ is bounded by cone ratios: -γ/w_M < u₁ < 1/w_L
- u₂ is bounded below by g(u₁), not above

This means:
- **κ → ∞**: The image approaches (but never reaches) the full disk
- **κ finite**: The image is a proper subset, with hue-dependent max saturation
- **"100% coverage" is impossible** for any finite κ

**Area fraction** (not "coverage"): For default θ with κ=1, approximately 48% of disk area is attainable. For κ=2, approximately 65%. Neither reaches 100%.

Avoid the term "coverage" which conflates:
1. Area fraction of D attainable from all LMS
2. Area fraction of D reached by a specific display gamut (sRGB, P3, etc.)
3. Percentage of colors that map without saturation

These are different quantities. Use precise language in all contexts.

### 7.5 Parameters Not Calibrated

The default θ values are **mathematical placeholders**, not psychophysically validated:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| w_L, w_M | 1.0, 1.0 | Equal luminance weighting |
| γ | 1.0 | L/M balance for achromatic |
| β | 0.5 | S/(L+M) balance for achromatic |
| ε | 0.01 | Luminance floor |
| κ | 1.0 | Compression gain |

**Whitepoint calibration**: Use `Theta.from_whitepoint(L, M, S)` to set γ, β so that a chosen neutral maps to the origin.

### 7.6 Reconstruction Limitations

1. **Positivity not guaranteed**: `reconstruct_lms(v, Y)` may produce negative L, M, or S for chromaticities outside the attainable region at the given luminance.

2. **Saturation regime**: Reconstruction is numerically unreliable when the original κ||u|| exceeded ~18.

### 7.7 Perceptual Validity Not Established

This implementation is a **faithful realization of Part I mathematics**. It does **NOT** establish:
- Hue angles match human hue perception
- Hilbert distance correlates with discrimination thresholds (JNDs)
- Entropy-based saturation correlates with perceived saturation
- Default parameters are meaningful for real observers

Perceptual validation requires Part II calibration work.

### 7.8 Single Observer Model

No inter-observer variability or chromatic adaptation states are modeled. The mapping is deterministic for a fixed θ.

### 7.9 Pullback Metric Degeneracy

The pullback metric G_LMS(x) = Jᵀ G_D(Φ(x)) J is **rank-deficient** by design:

**Why rank ≤ 2?**
- Φ_θ maps ℝ³ → ℝ² (3D LMS space to 2D Bloch disk)
- The Jacobian J is 2×3, so Jᵀ G_D J is at most rank 2

**For ε = 0 (exact scale invariance):**
- The scale direction x is in the null space of G_LMS
- Infinitesimal changes δx ∝ x produce δv = 0 (no chromatic change)
- This is mathematically correct: scaling doesn't change chromaticity

**For ε > 0:**
- The null space is only approximate
- G_LMS has two dominant eigenvalues and one near-zero eigenvalue
- The small eigenvalue corresponds to a direction "mostly" scaling

**Implications for discrimination ellipsoids:**
- In LMS space, the ellipsoid {δx : δxᵀ G_LMS δx ≤ 1} is infinite along the null direction
- For visualization, work in a 2D subspace (e.g., constant-Y chromaticity plane)
- Or compute ellipses directly in v-space using the Klein metric

**Code example:**
```python
G_LMS = pullback_metric_lms(lms, theta)
eigenvalues = np.linalg.eigvalsh(G_LMS)
# eigenvalues[0] ~ eigenvalues[1] >> eigenvalues[2] ≈ 0
```

---

## 8. Experiments and Results

This section documents the key experimental validations performed and their quantitative results.

### 8.1 Figure Index

| Figure | Script | Key Result |
|--------|--------|------------|
| `roundtrip_error_profile.png` | `roundtrip_precision_profile.py` | Error < 10⁻⁸ for κ||u|| < 11.7 |
| `lms_roundtrip_error_profile.png` | `roundtrip_precision_profile.py` | Full pipeline error profile |
| `gamut_boundary_analysis.png` | `gamut_boundary_analysis.py` | Area fraction ~61% (κ=1, D65) |
| `kappa_sensitivity.png` | `gamut_boundary_analysis.py` | Area varies with κ |
| `discrimination_ellipses.png` | `metric_analysis.py` | Metric diverges at boundary |
| `sensitivity_heatmap.png` | `metric_analysis.py` | Jacobian norm across disk |
| `distance_comparison.png` | `metric_analysis.py` | Hilbert~Bures correlation r≈0.98 |
| `srgb_grid_analysis.png` | `srgb_grid_analysis.py` | max ||u|| ≈ 6.06 for sRGB |
| `srgb_analysis.png` | `wide_gamut_analysis.py` | sRGB gamut in disk |
| `display_p3_analysis.png` | `wide_gamut_analysis.py` | Display P3 gamut in disk |
| `rec.2020_analysis.png` | `wide_gamut_analysis.py` | Rec.2020 gamut in disk |
| `gamut_comparison.png` | `wide_gamut_analysis.py` | Gamut size comparison |
| `image_demo_synthetic.png` | `image_hue_saturation_demo.py` | Hue/saturation decomposition |
| `image_demo_statistics.png` | `image_hue_saturation_demo.py` | Diagnostic distributions |
| `image_demo_hsv_wheel.png` | `image_hue_saturation_demo.py` | HSV wheel test pattern |
| `realistic_colors_demo.png` | `demo_realistic_colors.py` | Real LMS samples mapped |
| `bloch_scatter.png` | `chromabloch.demo` | Random LMS in Bloch disk |
| `saturation_hue_wheel.png` | `chromabloch.demo` | Saturation by hue angle |
| `u_region.png` | `chromabloch.demo` | Pre-compression u-space |

### 8.2 Plot Card: Roundtrip Error Profile

**File:** `examples/roundtrip_error_profile.png`, `examples/compression_roundtrip_profile.json`

**Purpose:** Validate the measured thresholds for reconstruction reliability.

**Computation:**
1. Generate κ||u|| values from 0.1 to 25 (log-spaced, n=1000)
2. For each x = κ||u||, compute `u → T_κ(u) → T_κ⁻¹(T_κ(u)) → u'`
3. Measure relative error `|u - u'| / |u|`

**Key Results (from JSON):**
- R_MAX = 1 − 10⁻¹⁵ (disk interior clamp)
- Invertibility cap: atanh(R_MAX) ≈ 17.6
- Error < 10⁻¹² for x < 7.2
- Error < 10⁻⁸ for x < 11.7
- Error < 10⁻⁶ for x < 15.8
- tanh(x) = 1.0 exactly at x ≈ 19.1 (float64 saturation)

**Interpretation:** The disk clamp (R_MAX) sets a hard invertibility cap at x ≈ 17.6. Beyond this, arctanh cannot recover the original value. The warning threshold (15) provides margin for ≈10⁻⁶ precision.

### 8.3 Plot Card: Attainable Region Analysis

**File:** `examples/gamut_boundary_analysis.png`, `examples/attainable_area_stats_*.json`

**Purpose:** Visualize non-surjectivity of Φ_θ.

**Computation:**
1. For each hue angle φ (720 samples), compute max ||u|| in attainable region
2. Map boundary to v-space via T_κ
3. Compute area fraction using polar integration AND grid counting

**Key Results (D65-calibrated θ, κ=1):**
- Area fraction (polar): 0.6115
- Area fraction (grid): 0.6128
- Discrepancy: 0.0012 < 0.02 ✓
- Max ||v|| varies by hue: 0.50 (yellow) to 1.0 (blue)

**Interpretation:** With D65 calibration, ~61% of the Bloch disk is attainable. The unreachable region (~39%) is a fundamental geometric constraint from the positivity of LMS, not a numerical limitation.

### 8.4 Plot Card: Discrimination Ellipses

**File:** `examples/discrimination_ellipses.png`

**Purpose:** Visualize the Klein metric structure on the Bloch disk.

**Computation:**
1. Sample 7×7 grid of points with ||v|| < 0.9
2. At each point, compute Klein metric tensor G(v)
3. Visualize unit metric ellipse (directions where ds² = 1)

**Key Results:**
- At origin: metric is Euclidean (circular ellipses)
- Near boundary: metric diverges (ellipses shrink → finer discrimination)
- √det(G) = (1−||v||²)^(-3/2): at ||v||=0.9, this is ~12× larger than at origin

**Reference formula:** For the Klein metric, det(G) = (1−r²)^(-3):
- At r=0: √det(G) = 1
- At r=0.9: √det(G) = (0.19)^(-3/2) ≈ 12.1
- At r=0.99: √det(G) = (0.0199)^(-3/2) ≈ 355

**Interpretation:** Colors near the boundary (high saturation) have finer discrimination than achromatic colors. This is a prediction from the hyperbolic geometry that could be tested against psychophysical data.

### 8.5 Plot Card: Distance Comparison

**File:** `examples/distance_comparison.png`, `examples/distance_comparison.stats.json`

**Purpose:** Compare quantum distance measures with Hilbert distance.

**Computation:**
1. Sample 500 random point pairs in disk (uniform in polar: r ∈ [0.1, 0.9], θ ∈ [-π, π])
2. Compute Hilbert, trace, Bures distance, Bures angle, and Euclidean
3. Compute Pearson and Spearman correlations

**Key Results (from `distance_comparison.stats.json` and console output):**

| Distance     | Pearson r | Spearman r |
|--------------|-----------|------------|
| Trace        | 0.9626    | 0.9564     |
| Bures        | 0.9789    | 0.9775     |
| Bures angle  | 0.9794    | 0.9775     |
| Euclidean    | 0.9626    | 0.9564     |

**Interpretation:** All quantum distances correlate strongly with Hilbert distance. Bures distance and Bures angle track Hilbert most closely (r ≈ 0.98). Trace and Euclidean correlations are slightly lower (r ≈ 0.96).

**Important notes:**

1. **Trace ≡ Euclidean (sanity check):** For qubit/rebit density matrices, trace distance satisfies:
   
   D_trace(ρ(v₁), ρ(v₂)) = ½||v₁ − v₂||
   
   Therefore "Trace vs Hilbert" and "Euclidean vs Hilbert" are the same comparison (up to a factor of 2). The identical correlations confirm this identity holds in the implementation.

2. **Bures distance and Bures angle are monotone transforms of fidelity.** Both Bures distance (D_B = √(2(1−√F))) and Bures angle (θ_B = arccos(√F)) are derived from the same fidelity F. Their Spearman correlations match because Spearman is invariant to monotone transforms.

### 8.6 Wide-Gamut Safety Analysis

**Files:** `examples/srgb_grid_analysis.png`, `examples/display_p3_analysis.png`, `examples/rec.2020_analysis.png`, `examples/gamut_comparison.png`

**Purpose:** Validate that standard color spaces stay within safe κ||u|| bounds.

**Key Results (from `*_metadata.json`):**

| Color Space | Max ||u|| | Max ||v|| (κ=1) | Rec. κ (tol=1e-8) | Safe? |
|-------------|-----------|-----------------|-------------------|-------|
| sRGB        | 6.06      | ~1.0            | 1.71              | ✓     |
| Display P3  | 6.11      | ~1.0            | 1.70              | ✓     |
| Rec.2020    | 8.72      | ~1.0            | 1.19              | ✓     |

**Note:** With κ=1, max κ||u|| = max ||u||. All values are well below the tol=1e-8 threshold (11.7) and the invertibility cap (~17.6).

**κ recommendation policy:**
- `suggest_kappa_for_max_u_norm(max_u, tol=1e-8, safety=0.9)` targets κ||u|| < 0.9 × 11.7 = 10.5
- For sRGB (max ||u|| ≈ 6.06): κ_rec ≈ 10.5/6.06 ≈ 1.73
- This ensures reconstruction error < 10⁻⁸ with 10% safety margin

**Interpretation:** All three standard color spaces produce max ||u|| < 9, which is safe for κ=1. The recommended κ (from `suggest_kappa_for_max_u_norm`) ensures max κ||u|| < 10.5 for all gamuts, guaranteeing reconstruction tolerance below 10⁻⁸.

### 8.7 Plot Card: Image Decomposition Demo

**Files:** `examples/image_demo_synthetic.png`, `examples/image_demo_statistics.png`, `examples/image_demo_hsv_wheel.png`

**Script:** `examples/image_hue_saturation_demo.py`

**Purpose:** Demonstrate the mapping on pixel images, extracting hue and saturation channels.

**Test images:**
1. **HSV wheel** (image_demo_hsv_wheel.png): Synthetic HSV color wheel (256×256), serves as ground truth
2. **Synthetic gradient** (image_demo_synthetic.png): RGB gradients and color bars

**θ parameters:** D65-calibrated, ε=0.01, κ=1.0

**Computation:**
1. Load RGB image → linearize (sRGB EOTF) → XYZ → LMS (HPE)
2. Apply Φ_θ with image-mode (`epsilon > 0`, black pixels handled via ε)
3. Extract hue = atan2(v₂, v₁), saturation = 1 - S(ρ(v))
4. Display original, hue wheel, and saturation map

**Key Results:**
- Hue varies smoothly around the color wheel (confirms opponent encoding)
- Saturation peaks at fully saturated colors, drops to 0 at white/gray
- Statistics panel shows ||v|| and saturation distributions

**Interpretation:** The mapping produces perceptually meaningful hue and saturation channels. The HSV wheel test confirms that opponent channels correctly encode chromatic content.

### 8.8 Plot Card: Sensitivity Heatmap

**File:** `examples/sensitivity_heatmap.png`

**Script:** `examples/metric_analysis.py`

**Purpose:** Visualize how mapping sensitivity varies across the Bloch disk.

**Computation:**
1. Create 50×50 grid over Bloch disk (||v|| < 0.95)
2. At each point, compute det(G_Klein)^(1/2) = (1 - ||v||²)^(-3/2)
3. Color by log₁₀(√det(G))

**Key Results:**
- At center (v=0): √det(G) = 1 (unit sensitivity)
- At ||v|| = 0.9: √det(G) ≈ 12
- At ||v|| = 0.95: √det(G) ≈ 31

**Interpretation:** Sensitivity to small perturbations increases dramatically near the boundary. This is the hyperbolic geometry's prediction: saturated colors have finer discrimination than achromatic colors.

### 8.9 Plot Card: Demo Artifacts

**Files:** `results/<timestamp>/plots/bloch_scatter.png`, `saturation_hue_wheel.png`, `u_region.png`

**Script:** `python -m chromabloch.demo`

**Purpose:** Quick validation of the full pipeline with random LMS samples.

**Computation:**
1. Generate 1000 random LMS samples (log-uniform)
2. Apply Φ_θ with D65 calibration
3. Compute densities, entropies, saturation

**Panels:**
- **bloch_scatter.png**: v values colored by hue
- **saturation_hue_wheel.png**: Saturation vs hue angle (polar)
- **u_region.png**: Pre-compression u = (O₁/Y, O₂/Y) space

**Interpretation:** Confirms the mapping produces valid Bloch vectors (||v|| < 1) and that saturation correlates with distance from origin.

### 8.10 Reproducibility Manifest

All experiments can be reproduced with:

```bash
cd chromabloch
python examples/run_all_figures.py
```

This generates `examples/manifest.json` containing:
- Git commit hash
- Python version
- NumPy version
- Timestamp
- θ parameters
- List of generated files with checksums

---

## 9. Next Steps for Part II

1. **Parameter calibration**: Fit θ to MacAdam ellipse data or modern color-difference datasets.

2. **LMS conversion selection**: Choose and validate an XYZ→LMS matrix (e.g., CAT02, Hunt-Pointer-Estevez).

3. **Observer-specific domains**: Estimate convex subsets Ω ⊂ D with Hilbert/Finsler geometry matched to discrimination ellipses.

4. **Context effects**: Model background-dependent transformations using structured state maps (channels).

5. **Psychophysical evaluation**: Compare predicted distances against human discrimination thresholds.

6. **Induced metric analysis**: Use Jacobian to compute pullback metric on LMS space (implemented in this Phase).

---

## 10. References

1. Part I derivations v8.tex (this repository)
2. Berthier, M. (2020). Geometry of color perception. Part 2: perceived colors from real quantum states and Hering's rebit. *J. Math. Neurosci.*, 10:14.
3. Provenzi, E. (2020). Geometry of color perception. Part 1: structures and metrics of a homogeneous color space. *J. Math. Neurosci.*, 10:7.
4. Prencipe, N. (2022). *Théorie et applications d'une nouvelle formulation de l'espace des couleurs perçues*. PhD thesis, Univ. La Rochelle.
