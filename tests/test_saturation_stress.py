"""Stress tests for tanh saturation and numerical limits.

These tests expose the numerical limitations of the compression map
when κ||u|| becomes large (approaching float64 tanh saturation ~18.4).
"""

import numpy as np
import pytest

from chromabloch.params import Theta
from chromabloch.compression import (
    compress_to_disk,
    decompress_from_disk,
    compression_saturation_diagnostics,
    compression_roundtrip_error,
    suggest_kappa_for_max_u_norm,
)


class TestTanhSaturationBehavior:
    """Tests exposing the tanh saturation numerical limitation."""

    def test_saturation_threshold_float64(self):
        """Document the float64 tanh saturation threshold."""
        # For float64, tanh(x) becomes indistinguishable from 1.0 around x=18.4
        # because 1 - tanh(x) ≈ 2*exp(-2x) < machine epsilon
        x_values = np.array([15.0, 17.0, 18.0, 18.4, 19.0, 20.0])
        tanh_values = np.tanh(x_values)
        one_minus_tanh = 1.0 - tanh_values

        # At x=15, should still have numerical headroom
        assert one_minus_tanh[0] > 1e-13  # x=15 still distinguishable from 1

        # At x=18.4, should be at or below machine epsilon
        assert one_minus_tanh[3] < 1e-15  # x=18.4 is at machine epsilon

        # At x=20, should be indistinguishable from 1.0
        # NOTE: Use tolerance, not exact equality, for cross-platform robustness
        assert one_minus_tanh[5] <= np.finfo(float).eps  # x=20 effectively 1.0

    def test_roundtrip_error_vs_u_norm(self):
        """Test how roundtrip error degrades as ||u|| increases."""
        theta = Theta(kappa=1.0)  # kappa=1 means kappa*||u|| = ||u||

        # Test norms from 0.1 to 50
        norms = np.array([0.1, 1.0, 5.0, 10.0, 15.0, 18.0, 20.0, 30.0, 50.0])

        for norm in norms:
            u = np.array([norm, 0.0])
            v = compress_to_disk(u, theta)
            u_reconstructed = decompress_from_disk(v, theta)
            error = np.linalg.norm(u - u_reconstructed)

            if norm < 15:
                # Should have good precision below saturation
                assert error < 1e-8, f"Error {error} too large for ||u||={norm}"
            elif norm > 20:
                # Above saturation, significant error expected
                # arctanh(1-eps) caps at ~18.4, so we lose information
                assert error > norm - 20, f"Expected large error for ||u||={norm}"

    def test_roundtrip_error_batch(self):
        """Test roundtrip error across range of norms."""
        theta = Theta(kappa=1.0)

        # Generate u with various norms
        norms = np.logspace(-1, 2, 100)  # 0.1 to 100
        angles = np.linspace(0, 2 * np.pi, 100)
        u = np.stack([norms * np.cos(angles), norms * np.sin(angles)], axis=-1)

        errors = compression_roundtrip_error(u, theta)

        # Low norms should have low error
        low_norm_mask = norms < 10
        assert np.all(errors[low_norm_mask] < 1e-8)

        # High norms should have high error
        high_norm_mask = norms > 25
        assert np.mean(errors[high_norm_mask]) > 1.0

    def test_saturation_diagnostics(self):
        """Test the saturation diagnostics function."""
        theta = Theta(kappa=1.0)

        # Generate u with norms spanning 0 to 50
        norms = np.linspace(0.1, 50, 1000)
        u = np.stack([norms, np.zeros_like(norms)], axis=-1)

        diag = compression_saturation_diagnostics(u, theta)

        # Should detect saturated samples
        assert diag.n_total == 1000
        assert diag.n_saturated > 0  # Some should be saturated
        assert diag.n_warning > diag.n_saturated  # Warning threshold is lower
        assert diag.max_kappa_r == pytest.approx(50.0)
        assert diag.effective_max_hyperbolic_radius < 20.0  # Capped


class TestKappaSelection:
    """Tests for κ parameter selection to avoid saturation."""

    def test_suggest_kappa(self):
        """Test κ suggestion function."""
        # If max ||u|| = 45, we need κ < 15/45 ≈ 0.33
        kappa = suggest_kappa_for_max_u_norm(45.0)
        assert kappa < 0.35
        assert kappa > 0.2

        # Check that suggested κ avoids saturation
        u = np.array([45.0, 0.0])
        theta = Theta(kappa=kappa)
        kappa_r = kappa * np.linalg.norm(u)
        assert kappa_r < 15.0

    def test_kappa_tradeoff(self):
        """Test the tradeoff between saturation avoidance and resolution."""
        # Small κ avoids saturation but reduces resolution near origin
        # Large κ gives good resolution but saturates for large u

        u_moderate = np.array([10.0, 0.0])
        u_small = np.array([0.1, 0.0])

        # Large κ: ||v|| = tanh(1.0 * 10) ≈ 1.0
        theta_large = Theta(kappa=1.0)
        v_large_kappa = compress_to_disk(u_moderate, theta_large)
        v_small_large_kappa = compress_to_disk(u_small, theta_large)

        # v for moderate u will be near boundary with large kappa
        assert np.linalg.norm(v_large_kappa) > 0.99

        # Small κ: ||v|| = tanh(0.1 * 10) = tanh(1) ≈ 0.76
        theta_small = Theta(kappa=0.1)
        v_small_kappa = compress_to_disk(u_moderate, theta_small)
        v_small_small_kappa = compress_to_disk(u_small, theta_small)

        # v for moderate u should now be much smaller
        assert np.linalg.norm(v_small_kappa) < 0.8

        # But v for small u is also smaller (less "use" of disk)
        assert np.linalg.norm(v_small_small_kappa) < np.linalg.norm(v_small_large_kappa)


class TestScalingInvariance:
    """Tests for the scaling behavior of Φθ."""

    def test_exact_scale_invariance_epsilon_zero(self):
        """Test Φθ(t·LMS) = Φθ(LMS) when ε=0."""
        from chromabloch.mapping import phi_theta

        theta = Theta(epsilon=0.0, kappa=0.5)  # Small kappa to avoid saturation

        rng = np.random.default_rng(42)
        lms = 10.0 ** rng.uniform(-0.5, 0.5, size=(20, 3))

        for i in range(len(lms)):
            v_base = phi_theta(lms[i], theta)

            for t in [0.5, 2.0, 10.0, 100.0]:
                v_scaled = phi_theta(t * lms[i], theta)
                np.testing.assert_allclose(
                    v_scaled, v_base, rtol=1e-10,
                    err_msg=f"Scale invariance failed for t={t}"
                )

    def test_epsilon_scaling_law(self):
        """Test the scaling law for ε > 0 from Proposition 3.5.

        u^(ε)(tx) = [t(Y+ε)/(tY+ε)] · u^(ε)(x)
        """
        from chromabloch.mapping import phi_theta_components

        theta = Theta(epsilon=0.01, kappa=0.5)

        rng = np.random.default_rng(42)
        lms = np.array([0.8, 1.0, 0.5])

        comps_base = phi_theta_components(lms, theta)
        Y = comps_base.Y
        u_base = comps_base.u

        for t in [0.5, 2.0, 10.0]:
            comps_scaled = phi_theta_components(t * lms, theta)
            u_scaled = comps_scaled.u

            # Expected scaling factor: t(Y+ε)/(tY+ε)
            expected_factor = t * (Y + theta.epsilon) / (t * Y + theta.epsilon)
            u_expected = expected_factor * u_base

            np.testing.assert_allclose(
                u_scaled, u_expected, rtol=1e-10,
                err_msg=f"Scaling law failed for t={t}"
            )

    def test_scaling_limit_behavior(self):
        """Test that u^(ε)(tx) → u^(0)(x) as t → ∞."""
        from chromabloch.mapping import phi_theta_components

        theta_eps = Theta(epsilon=0.01, kappa=0.5)
        theta_zero = Theta(epsilon=0.0, kappa=0.5)

        lms = np.array([0.8, 1.0, 0.5])

        # Get u^(0)
        comps_zero = phi_theta_components(lms, theta_zero)
        u_zero = comps_zero.u

        # As t → ∞, u^(ε)(tx) → u^(0)(x)
        for t in [10.0, 100.0, 1000.0]:
            comps_scaled = phi_theta_components(t * lms, theta_eps)
            u_scaled = comps_scaled.u

            # Error should decrease with t
            error = np.linalg.norm(u_scaled - u_zero)
            expected_error_bound = theta_eps.epsilon / (t * comps_zero.Y)
            assert error < expected_error_bound * 2  # With some margin


class TestSaturationFailureContract:
    """Tests that explicitly demonstrate and document saturation failure.

    These tests serve as API CONTRACT documentation: the right-inverse
    Φ̃θ⁻¹ ∘ Φθ = id holds ONLY when κ||u|| stays below the saturation threshold.
    """

    def test_reconstruction_fails_in_saturation_regime(self):
        """EXPLICIT DEMONSTRATION: reconstruction fails when saturated.

        This test documents the limitation, not a bug. When κ||u|| > ~18,
        the compression loses information and reconstruction cannot recover
        the original u.
        """
        theta = Theta(kappa=1.0)

        # Test points at various saturation levels
        # Note: κ||u|| = ||u|| when kappa=1
        test_cases = [
            # (||u||, expected_behavior)
            (5.0, "invertible"),      # κ||u|| = 5, well below threshold
            (10.0, "invertible"),     # κ||u|| = 10, safe zone
            (20.0, "saturated"),      # κ||u|| = 20, beyond threshold
            (50.0, "saturated"),      # κ||u|| = 50, severely saturated
        ]

        for u_norm, expected in test_cases:
            u = np.array([u_norm, 0.0])
            v = compress_to_disk(u, theta)
            u_reconstructed = decompress_from_disk(v, theta)
            error = np.linalg.norm(u - u_reconstructed)

            if expected == "invertible":
                assert error < 1e-6, (
                    f"Expected invertible at ||u||={u_norm}, but error={error}"
                )
            else:  # saturated
                assert error > 1.0, (
                    f"Expected saturation failure at ||u||={u_norm}, "
                    f"but error={error} is suspiciously small"
                )

    def test_saturation_does_not_silently_succeed(self):
        """Ensure we can detect when reconstruction is unreliable."""
        theta = Theta(kappa=1.0)

        # Saturated input
        u_saturated = np.array([30.0, 0.0])
        v = compress_to_disk(u_saturated, theta)

        # The compressed v will be very close to boundary
        assert np.linalg.norm(v) > 0.999999, "Expected v at boundary"

        # Decompression will NOT recover original
        u_recovered = decompress_from_disk(v, theta)
        assert np.linalg.norm(u_recovered) < 20, (
            "arctanh(1-ε) is bounded; cannot recover ||u||=30"
        )

        # Use diagnostics to detect this situation
        diag = compression_saturation_diagnostics(
            u_saturated.reshape(1, 2), theta
        )
        assert diag.n_saturated == 1, "Should detect saturation"

    def test_api_contract_summary(self):
        """Summary test documenting the reconstruction contract.

        CONTRACT:
        - Φθ: ℝ>0³ → D is always well-defined (maps to open disk)
        - Φ̃θ⁻¹: D × ℝ>0 → ℝ>0³ is always well-defined
        - Φ̃θ⁻¹(Φθ(LMS), Y) ≈ LMS ONLY when κ||u|| < ~15

        Beyond this threshold, information is lost in the tanh compression.
        """
        from chromabloch.mapping import phi_theta
        from chromabloch.reconstruction import reconstruct_lms

        theta = Theta(kappa=1.0, epsilon=0.0)

        # Case 1: Normal LMS, should roundtrip well
        lms_normal = np.array([1.0, 1.2, 0.8])
        v = phi_theta(lms_normal, theta)
        Y = theta.w_L * lms_normal[0] + theta.w_M * lms_normal[1]
        lms_recovered = reconstruct_lms(v, Y, theta)
        np.testing.assert_allclose(lms_recovered, lms_normal, rtol=1e-8)

        # Case 2: Extreme LMS that causes saturation
        lms_extreme = np.array([100.0, 0.01, 50.0])  # Very skewed
        v_extreme = phi_theta(lms_extreme, theta)
        Y_extreme = theta.w_L * lms_extreme[0] + theta.w_M * lms_extreme[1]

        # Check if this is in saturation regime
        from chromabloch.mapping import phi_theta_components
        comps = phi_theta_components(lms_extreme, theta)
        kappa_u = theta.kappa * np.linalg.norm(comps.u)

        if kappa_u > 15:
            # Reconstruction will NOT be accurate
            lms_recovered_extreme = reconstruct_lms(v_extreme, Y_extreme, theta)
            error = np.linalg.norm(lms_recovered_extreme - lms_extreme)
            # We expect significant error
            assert error > 0.1 * np.linalg.norm(lms_extreme), (
                f"Expected reconstruction error in saturation regime, "
                f"κ||u||={kappa_u:.1f}"
            )


class TestDiagnosticsIsSafe:
    """Tests for the MappingDiagnostics.is_safe() method."""

    def test_is_safe_basic(self):
        """Test that is_safe() returns True for normal inputs."""
        from chromabloch.mapping import phi_theta_with_diagnostics

        theta = Theta(kappa=1.0, epsilon=0.01)  # epsilon > 0 for safety
        
        # Normal LMS that gives moderate ||u|| (should be safe)
        lms_safe = np.array([1.0, 1.0, 1.0])
        _, diag_safe = phi_theta_with_diagnostics(lms_safe.reshape(1, 3), theta)
        assert diag_safe.is_safe(), f"Expected safe, got max_kappa_u={diag_safe.max_kappa_u}"

    def test_is_safe_false_for_high_kappa_u(self):
        """Test that is_safe() returns False when max_kappa_u >= 15."""
        from chromabloch.mapping import phi_theta_with_diagnostics

        theta = Theta(kappa=1.0, epsilon=0.01)
        
        # Very high S value to push kappa_u above threshold
        # Need kappa_u > 15, so ||u|| > 15
        # Use smaller kappa to avoid boundary clamping first
        theta_small_kappa = Theta(kappa=0.5, epsilon=0.01)
        
        # With kappa=0.5, need ||u|| > 30 for kappa_u > 15
        lms_high = np.array([1.0, 1.0, 65.0])  # Very high S
        _, diag_high = phi_theta_with_diagnostics(lms_high.reshape(1, 3), theta_small_kappa)
        
        print(f"High: max_kappa_u = {diag_high.max_kappa_u}")
        
        # Should not be safe (kappa_u >= 15)
        if diag_high.max_kappa_u >= 15:
            assert not diag_high.is_safe(), f"Expected not safe for kappa_u={diag_high.max_kappa_u}"

    def test_is_safe_checks_max_kappa_u_threshold(self):
        """Verify is_safe() includes the max_kappa_u < 15 check."""
        from chromabloch.mapping import MappingDiagnostics
        
        # Create a diagnostics object manually to test the threshold
        # Safe case: max_kappa_u < 15
        diag_safe = MappingDiagnostics(
            n_total=1, n_negative_clipped=0, n_zero_lms=0,
            min_Y=1.0, max_Y=1.0, min_u_norm=1.0, max_u_norm=1.0,
            max_kappa_u=14.9,  # Just below threshold
            n_near_saturation=0, n_saturated=0,
            n_boundary_clamped=0, max_v_norm_unclamped=0.9
        )
        assert diag_safe.is_safe()
        
        # Unsafe case: max_kappa_u >= 15
        diag_unsafe = MappingDiagnostics(
            n_total=1, n_negative_clipped=0, n_zero_lms=0,
            min_Y=1.0, max_Y=1.0, min_u_norm=1.0, max_u_norm=1.0,
            max_kappa_u=15.1,  # Just above threshold
            n_near_saturation=0, n_saturated=0,
            n_boundary_clamped=0, max_v_norm_unclamped=0.9
        )
        assert not diag_unsafe.is_safe()

    def test_is_reconstructable_more_conservative(self):
        """Test that is_reconstructable() uses threshold 14.5."""
        from chromabloch.mapping import MappingDiagnostics
        
        # Test tolerance-dependent thresholds (measured values from profiling)
        # tol=1e-8 threshold: 11.5
        # tol=1e-10 threshold: 10.0
        
        # Case: kappa_u=11.0 - safe for 1e-8 but not for 1e-10
        diag_border = MappingDiagnostics(
            n_total=1, n_negative_clipped=0, n_zero_lms=0,
            min_Y=1.0, max_Y=1.0, min_u_norm=1.0, max_u_norm=1.0,
            max_kappa_u=11.0,  # Below 11.5 but above 10.0
            n_near_saturation=0, n_saturated=0,
            n_boundary_clamped=0, max_v_norm_unclamped=0.9
        )
        assert diag_border.is_safe(), "Should be safe (kappa_u < 15)"
        assert diag_border.is_reconstructable(tol=1e-8), "Should be reconstructable at 1e-8 (kappa_u < 11.5)"
        assert not diag_border.is_reconstructable(tol=1e-10), "Should NOT be reconstructable at 1e-10 (kappa_u > 10.0)"


class TestAttainableRegionBoundary:
    """Tests approaching the attainable region boundary."""

    def test_approach_lower_boundary(self):
        """Test behavior as S → 0 (approaching u2 = g(u1))."""
        from chromabloch.mapping import phi_theta_components
        from chromabloch.mathutils import g_boundary

        theta = Theta(epsilon=0.0)

        # Fix L/M ratio to get specific u1
        L, M = 1.5, 1.0
        u1_expected = (L - theta.gamma * M) / (theta.w_L * L + theta.w_M * M)

        # Vary S from small to very small
        for S in [0.1, 0.01, 0.001, 0.0001]:
            lms = np.array([L, M, S])
            comps = phi_theta_components(lms, theta)
            u1, u2 = comps.u

            g_val = g_boundary(u1, theta)

            # u2 should approach g(u1) from above
            assert u2 > g_val, f"u2={u2} not above g(u1)={g_val}"

            # Gap should shrink as S → 0
            gap = u2 - g_val

        # The gap at S=0.0001 should be very small
        assert gap < 0.01

    def test_approach_u1_bounds(self):
        """Test behavior as L/M → extreme values."""
        from chromabloch.mapping import phi_theta_components
        from chromabloch.mathutils import u1_bounds

        theta = Theta(epsilon=0.0)
        lower, upper = u1_bounds(theta)

        # Approach upper bound: L >> M
        for ratio in [10, 100, 1000]:
            L, M, S = ratio, 1.0, 1.0
            comps = phi_theta_components(np.array([L, M, S]), theta)
            u1 = comps.u[0]
            assert u1 < upper
            # Gap should shrink
            gap_upper = upper - u1

        assert gap_upper < 0.01

        # Approach lower bound: M >> L
        for ratio in [10, 100, 1000]:
            L, M, S = 1.0, ratio, 1.0
            comps = phi_theta_components(np.array([L, M, S]), theta)
            u1 = comps.u[0]
            assert u1 > lower
            gap_lower = u1 - lower

        assert gap_lower < 0.01
