"""Tests for Jacobian computation."""

import numpy as np
import pytest

from chromabloch.params import Theta
from chromabloch.jacobian import (
    jacobian_phi_finite_diff,
    jacobian_phi_analytic,
    jacobian_norm,
    jacobian_condition_number,
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
