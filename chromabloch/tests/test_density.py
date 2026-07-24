"""Tests for density matrix module."""

import numpy as np
import pytest

from chromabloch.density import (
    rho_of_v,
    bloch_from_rho,
    is_psd_2x2,
    trace_is_one,
    bloch_norm,
    von_neumann_entropy,
    saturation_sigma,
    hue_angle,
)


class TestRhoOfV:
    """Tests for rho_of_v function."""

    def test_achromatic_state(self):
        """Test rho(0,0) = I/2."""
        v = np.array([0.0, 0.0])
        rho = rho_of_v(v)

        expected = np.array([[0.5, 0.0], [0.0, 0.5]])
        np.testing.assert_allclose(rho, expected)

    def test_pure_state_v1_axis(self):
        """Test rho for v = (1, 0) (boundary)."""
        v = np.array([1.0, 0.0])
        rho = rho_of_v(v)

        expected = np.array([[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(rho, expected)

    def test_pure_state_v2_axis(self):
        """Test rho for v = (0, 1) (boundary)."""
        v = np.array([0.0, 1.0])
        rho = rho_of_v(v)

        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_allclose(rho, expected)

    def test_trace_is_one(self):
        """Test tr(rho) = 1 for random v in disk."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            angle = rng.uniform(-np.pi, np.pi)
            r = rng.uniform(0, 0.99)
            v = np.array([r * np.cos(angle), r * np.sin(angle)])
            rho = rho_of_v(v)

            assert np.isclose(np.trace(rho), 1.0)

    def test_determinant_formula(self):
        """Test det(rho) = (1 - ||v||^2) / 4."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            angle = rng.uniform(-np.pi, np.pi)
            r = rng.uniform(0, 0.99)
            v = np.array([r * np.cos(angle), r * np.sin(angle)])
            rho = rho_of_v(v)

            det = np.linalg.det(rho)
            expected = (1 - r**2) / 4

            np.testing.assert_allclose(det, expected, rtol=1e-10)

    def test_batch_processing(self):
        """Test batch processing."""
        v = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
        rho = rho_of_v(v)

        assert rho.shape == (3, 2, 2)


class TestBlochFromRho:
    """Tests for bloch_from_rho function."""

    def test_roundtrip(self):
        """Test v = bloch_from_rho(rho_of_v(v))."""
        rng = np.random.default_rng(42)

        for _ in range(20):
            angle = rng.uniform(-np.pi, np.pi)
            r = rng.uniform(0, 0.99)
            v = np.array([r * np.cos(angle), r * np.sin(angle)])

            rho = rho_of_v(v)
            v_recovered = bloch_from_rho(rho)

            np.testing.assert_allclose(v, v_recovered, rtol=1e-10)

    def test_factor_of_two_correct(self):
        """Test that v2 = 2*b, not v2 = b (common mistake)."""
        # rho = [[a, b], [b, c]] with a + c = 1
        # v1 = a - c, v2 = 2*b
        rho = np.array([[0.7, 0.2], [0.2, 0.3]])

        v = bloch_from_rho(rho)

        assert np.isclose(v[0], 0.7 - 0.3)  # v1 = a - c = 0.4
        assert np.isclose(v[1], 2 * 0.2)    # v2 = 2*b = 0.4

    def test_batch_processing(self):
        """Test batch processing."""
        rho = np.array([
            [[0.5, 0.0], [0.0, 0.5]],
            [[0.7, 0.2], [0.2, 0.3]],
        ])
        v = bloch_from_rho(rho)

        assert v.shape == (2, 2)


class TestValidationHelpers:
    """Tests for is_psd_2x2 and trace_is_one."""

    def test_is_psd_valid(self):
        """Test is_psd returns True for valid density matrices."""
        v = np.array([0.5, 0.3])
        rho = rho_of_v(v)

        assert is_psd_2x2(rho)

    def test_is_psd_invalid(self):
        """Test is_psd returns False for invalid matrices."""
        # Negative eigenvalue
        rho = np.array([[0.3, 0.5], [0.5, 0.7]])  # det = 0.21 - 0.25 < 0

        assert not is_psd_2x2(rho)

    def test_trace_is_one_valid(self):
        """Test trace_is_one for valid density matrices."""
        v = np.array([0.5, 0.3])
        rho = rho_of_v(v)

        assert trace_is_one(rho)

    def test_trace_is_one_invalid(self):
        """Test trace_is_one for invalid matrices."""
        rho = np.array([[0.6, 0.0], [0.0, 0.6]])  # trace = 1.2

        assert not trace_is_one(rho)


class TestBlochNorm:
    """Tests for bloch_norm function."""

    def test_basic(self):
        """Test basic norm computation."""
        v = np.array([3.0, 4.0])

        assert np.isclose(bloch_norm(v), 5.0)

    def test_batch(self):
        """Test batch processing."""
        v = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])

        norms = bloch_norm(v)

        np.testing.assert_allclose(norms, [5.0, 0.0, 1.0])


class TestVonNeumannEntropy:
    """Tests for von_neumann_entropy function."""

    def test_achromatic_max_entropy(self):
        """Test S(0,0) = 1 bit (max entropy)."""
        v = np.array([0.0, 0.0])

        S = von_neumann_entropy(v)

        np.testing.assert_allclose(S, 1.0, rtol=1e-10)

    def test_pure_state_zero_entropy(self):
        """Test S approaches 0 as ||v|| → 1."""
        v = np.array([0.9999, 0.0])

        S = von_neumann_entropy(v)

        assert S < 0.01  # Very low entropy

    def test_monotonic_in_radius(self):
        """Test that entropy decreases as ||v|| increases."""
        radii = np.linspace(0.1, 0.9, 10)
        v = np.stack([radii, np.zeros_like(radii)], axis=-1)

        S = von_neumann_entropy(v)

        # Should be strictly decreasing
        assert np.all(np.diff(S) < 0)

    def test_different_base(self):
        """Test entropy with different logarithm base - should be simple scaling."""
        # Use a non-zero v to test base conversion
        v = np.array([0.5, 0.0])

        S_base2 = von_neumann_entropy(v, base=2.0)
        S_base_e = von_neumann_entropy(v, base=np.e)

        # Base conversion: S_b = S_2 / log_2(b)
        # So S_e = S_2 / log_2(e)
        np.testing.assert_allclose(S_base_e, S_base2 / np.log2(np.e), rtol=1e-10)

        # Also test base 4
        S_base4 = von_neumann_entropy(v, base=4.0)
        np.testing.assert_allclose(S_base4, S_base2 / np.log2(4.0), rtol=1e-10)

    def test_max_entropy_values(self):
        """Test max entropy (achromatic) for different bases."""
        v_achrom = np.array([0.0, 0.0])

        # Max entropy = log_base(2) for a 2-level system
        S_base2 = von_neumann_entropy(v_achrom, base=2.0)
        S_base_e = von_neumann_entropy(v_achrom, base=np.e)

        np.testing.assert_allclose(S_base2, 1.0, rtol=1e-10)  # 1 bit
        np.testing.assert_allclose(S_base_e, np.log(2), rtol=1e-10)  # ln(2) nats


class TestSaturationSigma:
    """Tests for saturation_sigma function."""

    def test_achromatic_zero_saturation(self):
        """Test Sigma(0,0) = 0."""
        v = np.array([0.0, 0.0])

        sigma = saturation_sigma(v)

        np.testing.assert_allclose(sigma, 0.0, atol=1e-10)

    def test_pure_state_max_saturation(self):
        """Test Sigma approaches 1 as ||v|| → 1."""
        v = np.array([0.9999, 0.0])

        sigma = saturation_sigma(v)

        assert sigma > 0.99

    def test_complement_of_entropy(self):
        """Test Sigma = 1 - S."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            r = rng.uniform(0, 0.99)
            v = np.array([r, 0.0])

            S = von_neumann_entropy(v)
            sigma = saturation_sigma(v)

            np.testing.assert_allclose(sigma, 1.0 - S, rtol=1e-10)


class TestHueAngle:
    """Tests for hue_angle function."""

    def test_positive_v1_axis(self):
        """Test H(1,0) = 0."""
        v = np.array([1.0, 0.0])

        H = hue_angle(v)

        np.testing.assert_allclose(H, 0.0)

    def test_positive_v2_axis(self):
        """Test H(0,1) = pi/2."""
        v = np.array([0.0, 1.0])

        H = hue_angle(v)

        np.testing.assert_allclose(H, np.pi / 2)

    def test_negative_v1_axis(self):
        """Test H(-1,0) = pi."""
        v = np.array([-1.0, 0.0])

        H = hue_angle(v)

        np.testing.assert_allclose(np.abs(H), np.pi)

    def test_achromatic_nan(self):
        """Test H(0,0) = NaN."""
        v = np.array([0.0, 0.0])

        H = hue_angle(v)

        assert np.isnan(H)

    def test_batch_processing(self):
        """Test batch processing."""
        v = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

        H = hue_angle(v)

        assert H.shape == (3,)
