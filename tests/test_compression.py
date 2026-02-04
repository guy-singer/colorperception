"""Tests for radial compression module."""

import numpy as np
import pytest

from chromabloch.params import Theta
from chromabloch.compression import compress_to_disk, decompress_from_disk


class TestCompressToDisk:
    """Tests for compress_to_disk (T_kappa)."""

    def test_origin_maps_to_origin(self):
        """Test T_kappa(0) = 0."""
        theta = Theta.default()
        u = np.array([0.0, 0.0])

        v = compress_to_disk(u, theta)

        np.testing.assert_allclose(v, [0.0, 0.0])

    def test_output_inside_disk(self):
        """Test ||T_kappa(u)|| < 1 for any u."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        # Test various magnitudes
        for scale in [0.1, 1.0, 10.0, 100.0]:
            u = rng.standard_normal((100, 2)) * scale
            v = compress_to_disk(u, theta)
            norms = np.linalg.norm(v, axis=-1)

            assert np.all(norms < 1.0)

    def test_direction_preserved(self):
        """Test that T_kappa preserves direction."""
        theta = Theta.default()
        u = np.array([3.0, 4.0])

        v = compress_to_disk(u, theta)

        # Direction should be preserved (same unit vector)
        u_dir = u / np.linalg.norm(u)
        v_dir = v / np.linalg.norm(v)

        np.testing.assert_allclose(u_dir, v_dir, atol=1e-10)

    def test_near_zero_stability(self):
        """Test numerical stability for very small u."""
        theta = Theta.default()
        small_vals = [1e-10, 1e-12, 1e-15]

        for s in small_vals:
            u = np.array([s, 0.0])
            v = compress_to_disk(u, theta)

            # Should not be NaN or inf
            assert np.all(np.isfinite(v))
            # Should be approximately kappa * u for small u
            np.testing.assert_allclose(v[0], theta.kappa * s, rtol=1e-3)

    def test_batch_processing(self):
        """Test batch processing."""
        theta = Theta.default()
        u = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        v = compress_to_disk(u, theta)

        assert v.shape == (3, 2)
        assert np.all(np.linalg.norm(v, axis=-1) < 1.0)


class TestDecompressFromDisk:
    """Tests for decompress_from_disk (T_kappa^{-1})."""

    def test_origin_maps_to_origin(self):
        """Test T_kappa^{-1}(0) = 0."""
        theta = Theta.default()
        v = np.array([0.0, 0.0])

        u = decompress_from_disk(v, theta)

        np.testing.assert_allclose(u, [0.0, 0.0])

    def test_near_boundary_stability(self):
        """Test numerical stability for v near boundary."""
        theta = Theta.default()
        # Very close to boundary
        r = 0.9999999
        v = np.array([r, 0.0])

        u = decompress_from_disk(v, theta)

        assert np.all(np.isfinite(u))
        assert np.linalg.norm(u) > 0

    def test_near_zero_stability(self):
        """Test numerical stability for very small v."""
        theta = Theta.default()
        small_vals = [1e-10, 1e-12, 1e-15]

        for s in small_vals:
            v = np.array([s, 0.0])
            u = decompress_from_disk(v, theta)

            assert np.all(np.isfinite(u))


class TestRoundTrip:
    """Tests for compression/decompression round-trip."""

    def test_roundtrip_random(self):
        """Test T_kappa^{-1}(T_kappa(u)) = u."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        for _ in range(10):
            u = rng.standard_normal(2) * rng.uniform(0.1, 5.0)
            v = compress_to_disk(u, theta)
            u_reconstructed = decompress_from_disk(v, theta)

            np.testing.assert_allclose(u, u_reconstructed, rtol=1e-8)

    def test_roundtrip_batch(self):
        """Test round-trip with batch input."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        u = rng.standard_normal((100, 2)) * 2.0
        v = compress_to_disk(u, theta)
        u_reconstructed = decompress_from_disk(v, theta)

        np.testing.assert_allclose(u, u_reconstructed, rtol=1e-9)

    def test_inverse_roundtrip(self):
        """Test T_kappa(T_kappa^{-1}(v)) = v for v in disk."""
        theta = Theta.default()
        rng = np.random.default_rng(42)

        # Generate points inside disk
        angles = rng.uniform(-np.pi, np.pi, 100)
        radii = rng.uniform(0.01, 0.99, 100)
        v = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1)

        u = decompress_from_disk(v, theta)
        v_reconstructed = compress_to_disk(u, theta)

        np.testing.assert_allclose(v, v_reconstructed, rtol=1e-9)

    def test_different_kappa_values(self):
        """Test round-trip with different kappa values."""
        rng = np.random.default_rng(42)
        u = rng.standard_normal(2) * 2.0

        for kappa in [0.5, 1.0, 2.0, 5.0]:
            theta = Theta(kappa=kappa)
            v = compress_to_disk(u, theta)
            u_reconstructed = decompress_from_disk(v, theta)

            np.testing.assert_allclose(u, u_reconstructed, rtol=1e-8)
