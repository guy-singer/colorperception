"""Tests for attainable chromaticity region utilities."""

import numpy as np
import pytest

from chromabloch.params import Theta
from chromabloch.mathutils import (
    g_boundary,
    u1_bounds,
    in_attainable_region_u,
    sample_attainable_region,
    reconstruct_from_attainable,
)
from chromabloch.mapping import phi_theta_components


class TestGBoundary:
    """Tests for g_boundary function."""

    def test_formula(self):
        """Test g(u1) = -β/Δ * (γ + 1 + (w_M - w_L)*u1)."""
        theta = Theta.default()  # w_L=1, w_M=1, gamma=1, beta=0.5

        u1 = 0.0
        g = g_boundary(u1, theta)

        # Delta = 1*1 + 1 = 2
        # g(0) = -0.5/2 * (1 + 1 + 0) = -0.25 * 2 = -0.5
        expected = -theta.beta / theta.Delta * (theta.gamma + 1 + 0)

        np.testing.assert_allclose(g, expected)

    def test_affine_function(self):
        """Test that g is affine (linear + constant)."""
        theta = Theta.default()

        u1_vals = np.array([-0.5, 0.0, 0.5])
        g_vals = g_boundary(u1_vals, theta)

        # Check that g is affine: g(u1) = a*u1 + b
        # Fit line and check residuals
        coeffs = np.polyfit(u1_vals, g_vals, 1)
        fitted = np.polyval(coeffs, u1_vals)

        np.testing.assert_allclose(g_vals, fitted, rtol=1e-10)

    def test_negative_at_origin(self):
        """Test g(0) < 0 for valid parameters."""
        theta = Theta.default()

        g0 = g_boundary(0.0, theta)

        assert g0 < 0, "g(0) should be negative"


class TestU1Bounds:
    """Tests for u1_bounds function."""

    def test_bounds_values(self):
        """Test bound values (-γ/w_M, 1/w_L)."""
        theta = Theta(w_L=2.0, w_M=0.5, gamma=1.5, beta=0.5, epsilon=0.01, kappa=1.0)

        lower, upper = u1_bounds(theta)

        np.testing.assert_allclose(lower, -1.5 / 0.5)  # -3
        np.testing.assert_allclose(upper, 1 / 2.0)     # 0.5

    def test_bounds_open_interval(self):
        """Test that bounds define an open interval."""
        theta = Theta.default()

        lower, upper = u1_bounds(theta)

        assert lower < upper
        assert np.isfinite(lower)
        assert np.isfinite(upper)


class TestInAttainableRegionU:
    """Tests for in_attainable_region_u function."""

    def test_origin_in_region(self):
        """Test that (0, 0) might be in region (depends on g(0))."""
        theta = Theta.default()

        # g(0) = -0.5/2 * 2 = -0.5 < 0, so (0, 0) is in region
        u = np.array([0.0, 0.0])
        in_region = in_attainable_region_u(u, theta)

        assert in_region, "(0,0) should be in region when g(0) < 0"

    def test_point_above_boundary(self):
        """Test point clearly above boundary."""
        theta = Theta.default()

        u = np.array([0.0, 1.0])  # Well above g(0) = -0.5
        in_region = in_attainable_region_u(u, theta)

        assert in_region

    def test_point_below_boundary(self):
        """Test point clearly below boundary."""
        theta = Theta.default()

        u = np.array([0.0, -1.0])  # Below g(0) = -0.5
        in_region = in_attainable_region_u(u, theta)

        assert not in_region

    def test_point_outside_u1_bounds(self):
        """Test point outside u1 bounds."""
        theta = Theta.default()
        lower, upper = u1_bounds(theta)

        # Point with u1 too low
        u1 = np.array([lower - 0.5, 0.0])
        assert not in_attainable_region_u(u1, theta)

        # Point with u1 too high
        u2 = np.array([upper + 0.5, 0.0])
        assert not in_attainable_region_u(u2, theta)


class TestRandomLMSInRegion:
    """Tests that random positive LMS gives chromaticity in attainable region."""

    def test_random_lms_in_region(self):
        """Test that random positive LMS maps to attainable region (ε=0)."""
        theta_eps0 = Theta(
            w_L=1.0, w_M=1.0, gamma=1.0, beta=0.5,
            epsilon=0.0,  # Exact scale invariance
            kappa=1.0,
        )

        rng = np.random.default_rng(42)

        # Generate random positive LMS
        n_samples = 100
        lms = 10.0 ** rng.uniform(-1, 1, size=(n_samples, 3))

        # Compute chromaticity
        comps = phi_theta_components(lms, theta_eps0)
        u = comps.u

        # Check all in region (with small tolerance for numerics)
        in_region = in_attainable_region_u(u, theta_eps0, tol=1e-10)

        assert np.all(in_region), f"Not all LMS mapped to region: {in_region.sum()}/{n_samples}"

    def test_u1_in_bounds(self):
        """Test that u1 is always in (-γ/w_M, 1/w_L) for positive LMS."""
        theta_eps0 = Theta(
            w_L=1.0, w_M=1.0, gamma=1.0, beta=0.5,
            epsilon=0.0,
            kappa=1.0,
        )

        rng = np.random.default_rng(42)
        n_samples = 100
        lms = 10.0 ** rng.uniform(-2, 2, size=(n_samples, 3))

        comps = phi_theta_components(lms, theta_eps0)
        u1 = comps.u[..., 0]

        lower, upper = u1_bounds(theta_eps0)

        assert np.all(u1 > lower), f"Some u1 <= {lower}"
        assert np.all(u1 < upper), f"Some u1 >= {upper}"


class TestSufficiencyReconstruction:
    """Test the sufficiency construction from the proof."""

    def test_reconstruct_from_attainable_gives_correct_u(self):
        """Test that reconstructed LMS gives back the target chromaticity."""
        theta = Theta.default()
        theta_eps0 = Theta(
            w_L=theta.w_L, w_M=theta.w_M, gamma=theta.gamma, beta=theta.beta,
            epsilon=0.0, kappa=theta.kappa,
        )

        # Sample some points from attainable region
        rng = np.random.default_rng(42)
        lower, upper = u1_bounds(theta_eps0)

        for _ in range(10):
            u1 = rng.uniform(lower + 0.1, upper - 0.1)
            g_val = g_boundary(u1, theta_eps0)
            u2 = g_val + rng.uniform(0.1, 1.0)  # Above boundary

            # Reconstruct LMS
            lms = reconstruct_from_attainable(u1, u2, theta_eps0, M=1.0)

            # Check all positive
            assert np.all(lms > 0), f"Negative LMS: {lms}"

            # Compute u^(0) from reconstructed LMS
            comps = phi_theta_components(lms, theta_eps0)
            u_computed = comps.u

            np.testing.assert_allclose(u_computed[0], u1, rtol=1e-9)
            np.testing.assert_allclose(u_computed[1], u2, rtol=1e-9)


class TestSampleAttainableRegion:
    """Tests for sample_attainable_region function."""

    def test_samples_in_region(self):
        """Test that all samples are in the attainable region."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        u_samples = sample_attainable_region(theta, n_samples=100, rng=rng)

        in_region = in_attainable_region_u(u_samples, theta, tol=1e-10)

        assert np.all(in_region)

    def test_sample_shape(self):
        """Test output shape."""
        theta = Theta.default()

        u_samples = sample_attainable_region(theta, n_samples=50)

        assert u_samples.shape == (50, 2)
