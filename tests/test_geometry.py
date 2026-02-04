"""Tests for Hilbert/Klein disk geometry."""

import numpy as np
import pytest

from chromabloch.geometry import (
    hilbert_distance,
    hilbert_distance_from_origin,
    gamma_factor,
    klein_gyroadd,
    boundary_points_on_line,
)


class TestHilbertDistanceFromOrigin:
    """Tests for hilbert_distance_from_origin."""

    def test_formula_arctanh(self):
        """Test d_H(0, v) = arctanh(||v||)."""
        rng = np.random.default_rng(42)

        for _ in range(20):
            angle = rng.uniform(-np.pi, np.pi)
            r = rng.uniform(0.1, 0.99)
            v = np.array([r * np.cos(angle), r * np.sin(angle)])

            d = hilbert_distance_from_origin(v)
            expected = np.arctanh(r)

            np.testing.assert_allclose(d, expected, rtol=1e-10)

    def test_origin_zero_distance(self):
        """Test d_H(0, 0) = 0."""
        v = np.array([0.0, 0.0])

        d = hilbert_distance_from_origin(v)

        np.testing.assert_allclose(d, 0.0)

    def test_batch_processing(self):
        """Test batch computation."""
        v = np.array([[0.5, 0.0], [0.0, 0.3], [0.4, 0.4]])

        d = hilbert_distance_from_origin(v)

        assert d.shape == (3,)


class TestHilbertDistance:
    """Tests for hilbert_distance (Klein formula)."""

    def test_coincident_points(self):
        """Test d_H(p, p) = 0 (numerically)."""
        p = np.array([0.3, 0.4])

        d = hilbert_distance(p, p)

        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_from_origin_consistency(self):
        """Test that general formula matches special case for p=0."""
        rng = np.random.default_rng(42)

        for _ in range(20):
            angle = rng.uniform(-np.pi, np.pi)
            r = rng.uniform(0.1, 0.99)
            v = np.array([r * np.cos(angle), r * np.sin(angle)])
            origin = np.array([0.0, 0.0])

            d_general = hilbert_distance(origin, v)
            d_special = hilbert_distance_from_origin(v)

            np.testing.assert_allclose(d_general, d_special, rtol=1e-10)

    def test_symmetry(self):
        """Test d_H(p, q) = d_H(q, p)."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            p = rng.uniform(-0.8, 0.8, size=2)
            q = rng.uniform(-0.8, 0.8, size=2)

            d_pq = hilbert_distance(p, q)
            d_qp = hilbert_distance(q, p)

            np.testing.assert_allclose(d_pq, d_qp, rtol=1e-10)

    def test_triangle_inequality(self):
        """Test d_H(p, r) <= d_H(p, q) + d_H(q, r)."""
        rng = np.random.default_rng(42)

        for _ in range(20):
            p = rng.uniform(-0.7, 0.7, size=2)
            q = rng.uniform(-0.7, 0.7, size=2)
            r = rng.uniform(-0.7, 0.7, size=2)

            d_pr = hilbert_distance(p, r)
            d_pq = hilbert_distance(p, q)
            d_qr = hilbert_distance(q, r)

            assert d_pr <= d_pq + d_qr + 1e-10

    def test_batch_processing(self):
        """Test batch distance computation."""
        p = np.array([[0.1, 0.2], [0.3, 0.4]])
        q = np.array([[0.5, 0.1], [0.2, 0.6]])

        d = hilbert_distance(p, q)

        assert d.shape == (2,)

    def test_near_boundary_stability(self):
        """Test numerical stability for points near boundary."""
        p = np.array([0.0, 0.0])
        q = np.array([0.999, 0.0])

        d = hilbert_distance(p, q)

        assert np.isfinite(d)
        assert d > 0


class TestGammaFactor:
    """Tests for gamma_factor (Lorentz factor)."""

    def test_origin(self):
        """Test Gamma(0) = 1."""
        u = np.array([0.0, 0.0])

        Gamma = gamma_factor(u)

        np.testing.assert_allclose(Gamma, 1.0)

    def test_formula(self):
        """Test Gamma = 1/sqrt(1 - ||u||^2)."""
        u = np.array([0.6, 0.0])

        Gamma = gamma_factor(u)
        expected = 1.0 / np.sqrt(1 - 0.36)

        np.testing.assert_allclose(Gamma, expected)

    def test_increases_toward_boundary(self):
        """Test Gamma increases as ||u|| → 1."""
        radii = np.linspace(0.1, 0.99, 10)
        u = np.stack([radii, np.zeros_like(radii)], axis=-1)

        Gamma = gamma_factor(u)

        # Should be strictly increasing
        assert np.all(np.diff(Gamma) > 0)


class TestKleinGyroadd:
    """Tests for klein_gyroadd."""

    def test_identity_element(self):
        """Test 0 ⊕ v = v."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            # Generate v inside the disk
            angle = rng.uniform(-np.pi, np.pi)
            r = rng.uniform(0.1, 0.8)
            v = np.array([r * np.cos(angle), r * np.sin(angle)])
            zero = np.array([0.0, 0.0])

            result = klein_gyroadd(zero, v)

            np.testing.assert_allclose(result, v, rtol=1e-10)

    def test_inverse_property(self):
        """Test (-u) ⊕ u = 0."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            # Generate u inside the disk (radially)
            angle = rng.uniform(-np.pi, np.pi)
            r = rng.uniform(0.1, 0.8)
            u = np.array([r * np.cos(angle), r * np.sin(angle)])

            result = klein_gyroadd(-u, u)

            np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-10)

    def test_result_in_disk(self):
        """Test ||u ⊕ v|| < 1."""
        rng = np.random.default_rng(42)

        for _ in range(20):
            u = rng.uniform(-0.9, 0.9, size=2)
            v = rng.uniform(-0.9, 0.9, size=2)

            result = klein_gyroadd(u, v)

            assert np.linalg.norm(result) < 1.0

    def test_distance_via_gyroaddition(self):
        """Test d_H(u, v) = arctanh(||(-u) ⊕ v||)."""
        rng = np.random.default_rng(42)

        for _ in range(20):
            u = rng.uniform(-0.7, 0.7, size=2)
            v = rng.uniform(-0.7, 0.7, size=2)

            # Distance via formula
            d_formula = hilbert_distance(u, v)

            # Distance via gyroaddition
            diff = klein_gyroadd(-u, v)
            d_gyro = np.arctanh(np.linalg.norm(diff))

            np.testing.assert_allclose(d_formula, d_gyro, rtol=1e-8)

    def test_batch_processing(self):
        """Test batch gyroaddition."""
        u = np.array([[0.1, 0.2], [0.3, 0.4]])
        v = np.array([[0.5, 0.1], [0.2, 0.6]])

        result = klein_gyroadd(u, v)

        assert result.shape == (2, 2)


class TestBoundaryPointsOnLine:
    """Tests for boundary_points_on_line."""

    def test_origin_through_point(self):
        """Test line from origin through v."""
        v = np.array([0.5, 0.0])
        origin = np.array([0.0, 0.0])

        a_minus, a_plus = boundary_points_on_line(origin, v)

        # Should be at ±(1, 0)
        np.testing.assert_allclose(np.linalg.norm(a_minus), 1.0, rtol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(a_plus), 1.0, rtol=1e-10)

    def test_boundary_points_on_circle(self):
        """Test that boundary points lie on unit circle."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            p = rng.uniform(-0.7, 0.7, size=2)
            q = rng.uniform(-0.7, 0.7, size=2)

            a_minus, a_plus = boundary_points_on_line(p, q)

            np.testing.assert_allclose(np.linalg.norm(a_minus), 1.0, rtol=1e-8)
            np.testing.assert_allclose(np.linalg.norm(a_plus), 1.0, rtol=1e-8)

    def test_points_collinear(self):
        """Test that p, q, a_minus, a_plus are collinear."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            p = rng.uniform(-0.6, 0.6, size=2)
            q = rng.uniform(-0.6, 0.6, size=2)

            if np.linalg.norm(p - q) < 1e-6:
                continue

            a_minus, a_plus = boundary_points_on_line(p, q)

            # Check collinearity via cross product (2D)
            def cross_2d(a, b):
                return a[0] * b[1] - a[1] * b[0]

            pq = q - p
            pa_minus = a_minus - p
            pa_plus = a_plus - p

            np.testing.assert_allclose(cross_2d(pq, pa_minus), 0.0, atol=1e-8)
            np.testing.assert_allclose(cross_2d(pq, pa_plus), 0.0, atol=1e-8)
