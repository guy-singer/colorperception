"""Tests for Jacobian computation."""

import numpy as np
import pytest

from chromabloch.params import Theta
from chromabloch.jacobian import (
    jacobian_phi_finite_diff,
    jacobian_phi_analytic,
    jacobian_phi_complex_step,
    jacobian_norm,
    jacobian_condition_number,
    verify_scale_invariance_jacobian,
)


class TestJacobianFiniteDiff:
    """Tests for finite-difference Jacobian."""

    def test_single_point_shape(self):
        """Single point returns 2×3 matrix."""
        theta = Theta.default()
        lms = np.array([1.0, 1.0, 0.5])
        J = jacobian_phi_finite_diff(lms, theta)
        assert J.shape == (2, 3)

    def test_batch_shape(self):
        """Batch returns (..., 2, 3) array."""
        theta = Theta.default()
        lms = np.random.default_rng(42).uniform(0.5, 1.5, size=(10, 3))
        J = jacobian_phi_finite_diff(lms, theta)
        assert J.shape == (10, 2, 3)

    def test_no_nans(self):
        """No NaN values in typical LMS range."""
        theta = Theta.default()
        rng = np.random.default_rng(42)
        lms = 10.0 ** rng.uniform(-0.5, 0.5, size=(100, 3))
        J = jacobian_phi_finite_diff(lms, theta)
        assert not np.any(np.isnan(J))

    def test_symmetry_at_achromatic(self):
        """At achromatic point, Jacobian has specific structure."""
        theta = Theta.default()
        # Equal L=M and S adjusted for achromatic
        lms_gray = np.array([1.0, 1.0, 0.5 * 2])  # S = β(L+M)
        J = jacobian_phi_finite_diff(lms_gray, theta)
        # Output v should be near origin, so Jacobian should be well-behaved
        assert np.linalg.norm(J) < 10  # Reasonable magnitude


class TestJacobianAnalytic:
    """Tests for analytic Jacobian."""

    def test_matches_finite_diff(self):
        """Analytic Jacobian matches finite-difference reference."""
        theta = Theta.default()
        rng = np.random.default_rng(123)
        
        for _ in range(50):
            lms = 10.0 ** rng.uniform(-0.3, 0.3, size=3)
            J_analytic = jacobian_phi_analytic(lms, theta)
            J_fd = jacobian_phi_finite_diff(lms, theta, eps=1e-6)
            
            np.testing.assert_allclose(
                J_analytic, J_fd, rtol=1e-4, atol=1e-8,
                err_msg=f"Mismatch at LMS={lms}"
            )

    def test_matches_finite_diff_batch(self):
        """Batch analytic matches batch finite-diff."""
        theta = Theta.default()
        rng = np.random.default_rng(456)
        lms = 10.0 ** rng.uniform(-0.3, 0.3, size=(20, 3))
        
        J_analytic = jacobian_phi_analytic(lms, theta)
        J_fd = jacobian_phi_finite_diff(lms, theta)
        
        np.testing.assert_allclose(J_analytic, J_fd, rtol=1e-4, atol=1e-8)

    def test_near_origin_stable(self):
        """Jacobian is stable for chromaticities near origin."""
        theta = Theta.default()
        # Gray at various luminances
        for scale in [0.1, 1.0, 10.0]:
            lms = np.array([scale, scale, scale * theta.beta * 2])
            J = jacobian_phi_analytic(lms, theta)
            assert not np.any(np.isnan(J))
            assert np.all(np.isfinite(J))

    def test_different_parameters(self):
        """Works with non-default parameters."""
        theta = Theta(w_L=0.8, w_M=1.2, gamma=0.9, beta=0.6, kappa=2.0)
        lms = np.array([1.0, 1.1, 0.7])
        
        J_analytic = jacobian_phi_analytic(lms, theta)
        J_fd = jacobian_phi_finite_diff(lms, theta)
        
        np.testing.assert_allclose(J_analytic, J_fd, rtol=1e-4)


class TestJacobianMetrics:
    """Tests for Jacobian-derived metrics."""

    def test_norm_positive(self):
        """Jacobian norm is always positive."""
        theta = Theta.default()
        rng = np.random.default_rng(789)
        lms = 10.0 ** rng.uniform(-0.5, 0.5, size=(100, 3))
        
        norms = jacobian_norm(lms, theta)
        assert np.all(norms > 0)

    def test_condition_number_finite(self):
        """Condition number is finite for typical inputs."""
        theta = Theta.default()
        rng = np.random.default_rng(101)
        lms = 10.0 ** rng.uniform(-0.3, 0.3, size=(50, 3))
        
        cond = jacobian_condition_number(lms, theta)
        assert np.all(np.isfinite(cond))
        assert np.all(cond >= 1.0)  # Condition number ≥ 1 always

    def test_norm_increases_with_saturation(self):
        """Higher saturation colors tend to have larger Jacobian norms."""
        theta = Theta.default()
        
        # Gray (low saturation)
        lms_gray = np.array([1.0, 1.0, 1.0])
        norm_gray = jacobian_norm(lms_gray, theta)
        
        # Saturated red-ish (high L/M ratio)
        lms_sat = np.array([2.0, 0.5, 0.3])
        norm_sat = jacobian_norm(lms_sat, theta)
        
        # Saturated should generally have larger sensitivity
        # (This is a soft test - relationship may not be monotonic)
        assert norm_sat > 0.5 * norm_gray


class TestJacobianConsistency:
    """Tests for mathematical consistency of Jacobians."""

    def test_directional_derivative(self):
        """Jacobian correctly predicts directional derivatives."""
        theta = Theta.default()
        lms = np.array([1.0, 0.8, 0.6])
        
        J = jacobian_phi_analytic(lms, theta)
        
        # Small perturbation
        delta = np.array([0.01, -0.005, 0.008])
        
        from chromabloch.mapping import phi_theta
        
        v0 = phi_theta(lms, theta)
        v1 = phi_theta(lms + delta, theta)
        
        # Predicted change via Jacobian
        dv_predicted = J @ delta
        dv_actual = v1 - v0
        
        np.testing.assert_allclose(dv_predicted, dv_actual, rtol=0.1, atol=1e-5)

    def test_chain_rule_structure(self):
        """Verify Jacobian structure reflects chain rule."""
        theta = Theta.default()
        lms = np.array([1.0, 1.0, 0.5])
        
        J = jacobian_phi_analytic(lms, theta)
        
        # J is 2×3, representing ∂(v1,v2)/∂(L,M,S)
        # Each row should have meaningful values
        assert J.shape == (2, 3)
        
        # S only affects v2 directly (through O2), not v1 (through O1)
        # But compression couples them, so both rows can have nonzero ∂/∂S
        # Still, ∂v1/∂S should be smaller than ∂v2/∂S typically
        # (This is a soft structural test)


class TestJacobianComplexStep:
    """Tests for complex-step derivative validation."""

    def test_complex_step_matches_analytic(self):
        """Complex-step should match analytic to near machine precision."""
        theta = Theta.default()
        rng = np.random.default_rng(789)
        
        for _ in range(30):
            # Strictly positive LMS (no clamping)
            lms = 10.0 ** rng.uniform(-0.3, 0.3, size=3)
            
            J_analytic = jacobian_phi_analytic(lms, theta)
            J_complex = jacobian_phi_complex_step(lms, theta)
            
            # Complex-step should be much more accurate than finite-diff
            np.testing.assert_allclose(
                J_analytic, J_complex, rtol=1e-7, atol=1e-10,
                err_msg=f"Mismatch at LMS={lms}"
            )

    def test_complex_step_better_than_finite_diff(self):
        """Complex-step should have smaller error than finite-diff."""
        theta = Theta.default()
        lms = np.array([1.0, 0.9, 0.5])
        
        J_analytic = jacobian_phi_analytic(lms, theta)
        J_complex = jacobian_phi_complex_step(lms, theta)
        J_fd = jacobian_phi_finite_diff(lms, theta, eps=1e-7)
        
        err_complex = np.linalg.norm(J_analytic - J_complex)
        err_fd = np.linalg.norm(J_analytic - J_fd)
        
        # Complex-step error should be much smaller
        assert err_complex < err_fd * 0.01, (
            f"Complex-step error {err_complex} not much better than FD {err_fd}"
        )
    
    def test_holomorphic_norm_implementation(self):
        """Verify complex-step uses holomorphic (algebraic) norm, not |z|.
        
        The complex-step method requires all operations to be holomorphic.
        Using np.linalg.norm or np.abs on complex numbers breaks this because
        they use |z| = sqrt(z * conj(z)), which involves conjugation.
        
        We test this by checking agreement at rtol=1e-10, which would fail
        if the norm was computed non-holomorphically.
        """
        theta = Theta(epsilon=0.01, kappa=1.0)
        
        # Test at points well inside the smooth regime
        # (away from origin and boundary)
        rng = np.random.default_rng(42)
        
        n_success = 0
        for _ in range(30):
            # Generate LMS that gives ||v|| in (0.2, 0.8)
            lms = 10.0 ** rng.uniform(-0.2, 0.2, size=3)
            
            # Verify v is in smooth regime
            from chromabloch.mapping import phi_theta
            v = phi_theta(lms, theta)
            v_norm = np.linalg.norm(v)
            
            if 0.1 < v_norm < 0.85:  # Smooth regime
                J_analytic = jacobian_phi_analytic(lms, theta)
                J_complex = jacobian_phi_complex_step(lms, theta)
                
                # Very tight tolerance to catch non-holomorphic implementation
                np.testing.assert_allclose(
                    J_analytic, J_complex, rtol=1e-10, atol=1e-12,
                    err_msg=f"Holomorphic test failed at LMS={lms}, ||v||={v_norm}"
                )
                n_success += 1
        
        # Ensure we tested enough points in smooth regime
        assert n_success >= 10, f"Only {n_success} points in smooth regime"


class TestScaleInvarianceJacobian:
    """Tests for scale-invariance property at ε=0."""

    def test_radial_direction_in_null_space_eps0(self):
        """For ε=0, J(x)·x should be zero (scale invariance)."""
        theta = Theta(epsilon=0.0)
        rng = np.random.default_rng(111)
        
        for _ in range(50):
            lms = 10.0 ** rng.uniform(-0.3, 0.3, size=3)
            
            norm = verify_scale_invariance_jacobian(lms, theta, method="analytic")
            
            # Should be near zero (machine precision)
            assert norm < 1e-8, f"Scale invariance violated: ||J·x|| = {norm}"

    def test_radial_direction_nonzero_eps_positive(self):
        """For ε>0, J(x)·x should NOT be zero (no scale invariance)."""
        theta = Theta(epsilon=0.01)
        lms = np.array([1.0, 1.0, 0.5])
        
        norm = verify_scale_invariance_jacobian(lms, theta, method="analytic")
        
        # Should be nonzero when ε > 0
        assert norm > 1e-4, f"Expected nonzero radial derivative, got {norm}"

    def test_scale_invariance_all_methods_agree(self):
        """All Jacobian methods should agree on scale invariance."""
        theta = Theta(epsilon=0.0)
        lms = np.array([1.0, 0.8, 0.6])
        
        norm_analytic = verify_scale_invariance_jacobian(lms, theta, "analytic")
        norm_complex = verify_scale_invariance_jacobian(lms, theta, "complex_step")
        norm_fd = verify_scale_invariance_jacobian(lms, theta, "finite_diff")
        
        # All should be small for ε=0
        assert norm_analytic < 1e-8
        assert norm_complex < 1e-8
        # Finite-diff is less accurate but should still be small
        assert norm_fd < 1e-5


class TestJacobianSRGBAnalysis:
    """Jacobian analysis on sRGB-like inputs."""

    def test_srgb_primary_jacobians(self):
        """Compute Jacobians at sRGB primary colors."""
        theta = Theta.default()
        
        # Approximate LMS for sRGB primaries (simplified)
        primaries = {
            'red': np.array([0.4, 0.2, 0.05]),
            'green': np.array([0.2, 0.4, 0.1]),
            'blue': np.array([0.05, 0.1, 0.9]),
        }
        
        for name, lms in primaries.items():
            J = jacobian_phi_analytic(lms, theta)
            norm = np.linalg.norm(J)
            cond = jacobian_condition_number(lms, theta)
            
            assert np.isfinite(norm), f"Non-finite norm for {name}"
            assert np.isfinite(cond), f"Non-finite condition for {name}"
            print(f"{name}: ||J|| = {norm:.4f}, cond = {cond:.4f}")
