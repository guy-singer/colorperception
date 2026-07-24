"""Tests for mapping and reconstruction round-trip."""

import numpy as np
import pytest

from chromabloch.params import Theta
from chromabloch.mapping import phi_theta, phi_theta_components
from chromabloch.reconstruction import (
    reconstruct_lms,
    positivity_conditions,
    minimum_luminance_required,
    reconstruct_from_phi_roundtrip,
)
from chromabloch.opponent import opponent_transform
from chromabloch.density import bloch_norm


class TestPhiTheta:
    """Tests for phi_theta function."""

    def test_output_in_disk(self):
        """Test ||phi_theta(lms)|| < 1."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        lms = 10.0 ** rng.uniform(-1, 1, size=(100, 3))
        v = phi_theta(lms, theta)

        norms = bloch_norm(v)
        assert np.all(norms < 1.0)

    def test_achromatic_maps_to_origin(self):
        """Test achromatic locus maps to (0, 0)."""
        theta = Theta.default()

        # Achromatic: L = gamma*M, S = beta*(1+gamma)*M
        M = 1.0
        L = theta.gamma * M
        S = theta.beta * (1 + theta.gamma) * M
        lms = np.array([L, M, S])

        v = phi_theta(lms, theta)

        np.testing.assert_allclose(v, [0.0, 0.0], atol=1e-10)

    def test_batch_processing(self):
        """Test batch input processing."""
        theta = Theta.default()
        lms = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 0.5]])

        v = phi_theta(lms, theta)

        assert v.shape == (2, 2)

    def test_scaling_behavior(self):
        """Test approximate scale invariance for large Y."""
        theta = Theta.default()
        lms_base = np.array([0.8, 1.0, 0.5])

        v_base = phi_theta(lms_base, theta)

        # Scale up by large factor
        lms_scaled = lms_base * 100
        v_scaled = phi_theta(lms_scaled, theta)

        # Should be similar (approaching exact equality as scale → ∞)
        np.testing.assert_allclose(v_scaled, v_base, rtol=0.01)


class TestPhiThetaComponents:
    """Tests for phi_theta_components function."""

    def test_components_consistency(self):
        """Test that components are consistent."""
        theta = Theta.default()
        lms = np.array([0.8, 1.0, 0.5])

        comps = phi_theta_components(lms, theta)

        # Check Y computation
        Y_direct, _, _ = opponent_transform(lms, theta)
        np.testing.assert_allclose(comps.Y, Y_direct)

        # Check final v matches phi_theta
        v_direct = phi_theta(lms, theta)
        np.testing.assert_allclose(comps.v, v_direct)


class TestReconstructLms:
    """Tests for reconstruct_lms function."""

    def test_roundtrip_error(self):
        """Test reconstruction round-trip error is small."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        for _ in range(20):
            lms = 10.0 ** rng.uniform(-0.5, 0.5, size=3)

            # Forward
            v = phi_theta(lms, theta)
            Y, _, _ = opponent_transform(lms, theta)

            # Reconstruct
            lms_reconstructed = reconstruct_lms(v, Y, theta)

            np.testing.assert_allclose(lms, lms_reconstructed, rtol=1e-9)

    def test_roundtrip_batch(self):
        """Test batch reconstruction."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        lms = 10.0 ** rng.uniform(-0.5, 0.5, size=(50, 3))
        v = phi_theta(lms, theta)
        Y, _, _ = opponent_transform(lms, theta)

        lms_reconstructed = reconstruct_lms(v, Y, theta)

        np.testing.assert_allclose(lms, lms_reconstructed, rtol=1e-8)

    def test_roundtrip_helper(self):
        """Test the reconstruct_from_phi_roundtrip helper."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        lms = 10.0 ** rng.uniform(-0.5, 0.5, size=(20, 3))
        errors = reconstruct_from_phi_roundtrip(lms, theta)

        assert np.all(errors < 1e-8)


class TestAchromaticLocus:
    """Tests for achromatic locus behavior."""

    def test_achromatic_ray(self):
        """Test that achromatic ray maps to origin."""
        theta = Theta.default()

        for M in [0.1, 1.0, 10.0]:
            L = theta.gamma * M
            S = theta.beta * (1 + theta.gamma) * M
            lms = np.array([L, M, S])

            v = phi_theta(lms, theta)

            np.testing.assert_allclose(v, [0.0, 0.0], atol=1e-10)

    def test_achromatic_reconstruction(self):
        """Test reconstruction of achromatic point."""
        theta = Theta.default()

        v = np.array([0.0, 0.0])
        Y_target = 2.0

        lms = reconstruct_lms(v, Y_target, theta)

        # Verify it's achromatic
        v_check = phi_theta(lms, theta)
        np.testing.assert_allclose(v_check, [0.0, 0.0], atol=1e-10)


class TestPositivityConditions:
    """Tests for positivity_conditions function."""

    def test_valid_lms_all_positive(self):
        """Test that valid LMS gives all positive conditions."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        lms = 10.0 ** rng.uniform(-0.5, 0.5, size=3)
        v = phi_theta(lms, theta)
        Y, _, _ = opponent_transform(lms, theta)

        conds = positivity_conditions(v, Y, theta)

        assert conds["L_pos"]
        assert conds["M_pos"]
        assert conds["S_pos"]

    def test_positivity_margins(self):
        """Test that margins are positive for valid reconstruction."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        lms = 10.0 ** rng.uniform(-0.3, 0.3, size=(20, 3))
        v = phi_theta(lms, theta)
        Y, _, _ = opponent_transform(lms, theta)

        conds = positivity_conditions(v, Y, theta)

        assert np.all(conds["margin_L"] > 0)
        assert np.all(conds["margin_M"] > 0)
        assert np.all(conds["margin_S"] > 0)


class TestMinimumLuminanceRequired:
    """Tests for minimum_luminance_required function."""

    def test_minimum_luminance_allows_reconstruction(self):
        """Test that using Y >= Y_min gives valid reconstruction."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        # Get a v from valid LMS
        lms = np.array([0.8, 1.0, 0.5])
        v = phi_theta(lms, theta)

        Y_min = minimum_luminance_required(v, theta)

        # Y_min should be finite and reasonable
        assert np.isfinite(Y_min)
        assert Y_min >= 0

        # Using Y above Y_min should give positive reconstruction
        Y_test = Y_min + 0.1
        conds = positivity_conditions(v, Y_test, theta)

        assert conds["L_pos"]
        assert conds["M_pos"]
        assert conds["S_pos"]

    def test_minimum_luminance_batch(self):
        """Test batch minimum luminance computation."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        lms = 10.0 ** rng.uniform(-0.5, 0.5, size=(10, 3))
        v = phi_theta(lms, theta)

        Y_min = minimum_luminance_required(v, theta)

        assert Y_min.shape == (10,)
        assert np.all(np.isfinite(Y_min))


class TestOpponentAxes:
    """Tests for behavior on opponent axes."""

    def test_red_green_axis(self):
        """Test colors with O2 = 0 map to v1 axis."""
        theta = Theta.default()

        # O2 = S - beta*(L+M) = 0 => S = beta*(L+M)
        L, M = 1.5, 0.8
        S = theta.beta * (L + M)
        lms = np.array([L, M, S])

        v = phi_theta(lms, theta)

        # v2 should be ~0
        np.testing.assert_allclose(v[1], 0.0, atol=1e-10)
        # v1 should be nonzero (unless L = gamma*M)
        assert np.abs(v[0]) > 0.01

    def test_yellow_blue_axis(self):
        """Test colors with O1 = 0 map to v2 axis."""
        theta = Theta.default()

        # O1 = L - gamma*M = 0 => L = gamma*M
        M = 1.0
        L = theta.gamma * M
        S = 0.5  # Arbitrary, will determine v2 sign
        lms = np.array([L, M, S])

        v = phi_theta(lms, theta)

        # v1 should be ~0
        np.testing.assert_allclose(v[0], 0.0, atol=1e-10)
