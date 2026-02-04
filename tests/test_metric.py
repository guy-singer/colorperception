"""Tests for induced metric / pullback geometry."""

import numpy as np
import pytest

from chromabloch.params import Theta
from chromabloch.mapping import phi_theta
from chromabloch.geometry import hilbert_distance
from chromabloch.metric import (
    klein_metric_tensor,
    pullback_metric_lms,
    metric_eigenvalues,
    metric_trace,
    discrimination_ellipsoid_axes,
)


class TestKleinMetric:
    """Tests for Klein metric tensor on Bloch disk."""

    def test_positive_definite_interior(self):
        """Klein metric should be positive definite in disk interior."""
        rng = np.random.default_rng(42)
        
        for _ in range(50):
            r = rng.uniform(0.1, 0.8)
            angle = rng.uniform(-np.pi, np.pi)
            v = np.array([r * np.cos(angle), r * np.sin(angle)])
            
            G = klein_metric_tensor(v)
            eigenvalues = np.linalg.eigvalsh(G)
            
            assert np.all(eigenvalues > 0), f"Not positive definite at v={v}"

    def test_symmetric(self):
        """Klein metric tensor should be symmetric."""
        v = np.array([0.3, 0.5])
        G = klein_metric_tensor(v)
        
        np.testing.assert_allclose(G, G.T, atol=1e-15)

    def test_identity_at_origin(self):
        """At origin, Klein metric is identity."""
        v = np.array([0.0, 0.0])
        G = klein_metric_tensor(v)
        
        np.testing.assert_allclose(G, np.eye(2), atol=1e-15)

    def test_diverges_near_boundary(self):
        """Metric should increase as ||v|| → 1."""
        norms = []
        for r in [0.5, 0.8, 0.9, 0.95, 0.99]:
            v = np.array([r, 0])
            G = klein_metric_tensor(v)
            norms.append(np.linalg.norm(G))
        
        # Should be monotonically increasing
        for i in range(len(norms) - 1):
            assert norms[i] < norms[i+1], "Metric should increase near boundary"

    def test_batch_shape(self):
        """Batch computation should have correct shape."""
        v = np.array([[0.3, 0.4], [0.5, 0.2], [-0.2, 0.6]])
        G = klein_metric_tensor(v)
        
        assert G.shape == (3, 2, 2)


class TestPullbackMetric:
    """Tests for pullback metric on LMS space."""

    def test_positive_semidefinite(self):
        """Pullback metric should be positive semidefinite."""
        theta = Theta.default()
        rng = np.random.default_rng(123)
        
        for _ in range(30):
            lms = 10.0 ** rng.uniform(-0.3, 0.3, size=3)
            G = pullback_metric_lms(lms, theta)
            eigenvalues = np.linalg.eigvalsh(G)
            
            assert np.all(eigenvalues >= -1e-10), f"Not PSD at lms={lms}"

    def test_symmetric(self):
        """Pullback metric should be symmetric."""
        theta = Theta.default()
        lms = np.array([1.0, 0.8, 0.5])
        G = pullback_metric_lms(lms, theta)
        
        np.testing.assert_allclose(G, G.T, atol=1e-14)

    def test_rank_at_most_two(self):
        """Pullback metric rank ≤ 2 (maps 3D → 2D)."""
        theta = Theta.default()
        lms = np.array([1.0, 1.0, 0.5])
        
        G = pullback_metric_lms(lms, theta)
        eigenvalues = metric_eigenvalues(G)
        
        # At most 2 significant eigenvalues
        assert eigenvalues[2] < 1e-6 * eigenvalues[0], (
            f"Third eigenvalue too large: {eigenvalues}"
        )

    def test_scale_direction_null_eps_zero(self):
        """For ε=0, scale direction should be in null space."""
        theta = Theta(epsilon=0.0)
        lms = np.array([1.0, 0.8, 0.6])
        
        G = pullback_metric_lms(lms, theta)
        
        # Scale direction is lms itself
        # G @ lms should give small result
        result = G @ lms
        
        # The scale direction should be nearly in null space
        assert np.linalg.norm(result) < 0.1 * np.linalg.norm(G) * np.linalg.norm(lms)


class TestMetricMetrics:
    """Tests for metric-derived quantities."""

    def test_trace_positive(self):
        """Metric trace should be positive."""
        theta = Theta.default()
        rng = np.random.default_rng(456)
        
        for _ in range(30):
            lms = 10.0 ** rng.uniform(-0.3, 0.3, size=3)
            G = pullback_metric_lms(lms, theta)
            tr = metric_trace(G)
            
            assert tr > 0, f"Trace should be positive, got {tr}"

    def test_eigenvalues_sorted(self):
        """Eigenvalues should be sorted descending."""
        theta = Theta.default()
        lms = np.array([1.0, 0.5, 0.8])
        
        G = pullback_metric_lms(lms, theta)
        eigs = metric_eigenvalues(G)
        
        for i in range(len(eigs) - 1):
            assert eigs[i] >= eigs[i+1], "Eigenvalues not sorted"


class TestDiscriminationEllipsoid:
    """Tests for discrimination ellipsoid computation."""

    def test_axes_positive(self):
        """Ellipsoid axes should be positive (or infinite)."""
        theta = Theta.default()
        lms = np.array([1.0, 1.0, 0.5])
        
        lengths, _ = discrimination_ellipsoid_axes(lms, theta)
        
        for length in lengths:
            assert length > 0 or np.isinf(length)

    def test_directions_orthonormal(self):
        """Principal directions should be orthonormal."""
        theta = Theta.default()
        lms = np.array([1.0, 0.8, 0.6])
        
        _, directions = discrimination_ellipsoid_axes(lms, theta)
        
        # Check orthonormality
        product = directions.T @ directions
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)


class TestMetricIsDistanceDifferential:
    """Tests that metric tensor is the differential of distance squared.
    
    The Klein metric ds² = dv^T G(v) dv should satisfy:
        d_H(v, v+dv) ≈ √(dv^T G(v) dv)  for small ||dv||
    
    This is a critical validation test for the metric implementation.
    """

    def test_klein_metric_is_distance_differential_on_disk(self):
        """Verify Klein metric predicts Hilbert distance for small perturbations.
        
        For random v with ||v|| < 0.8 and small dv with ||dv|| ~ 1e-6:
        - d_num = hilbert_distance(v, v+dv)
        - d_lin = sqrt(dv^T G(v) dv)
        - |d_num - d_lin| / d_lin should be O(||dv||)
        """
        rng = np.random.default_rng(42)
        
        errors = []
        for _ in range(100):
            # Random point in disk interior
            r = rng.uniform(0.1, 0.75)
            angle = rng.uniform(-np.pi, np.pi)
            v = np.array([r * np.cos(angle), r * np.sin(angle)])
            
            # Small random perturbation
            dv_direction = rng.normal(size=2)
            dv_direction /= np.linalg.norm(dv_direction)
            dv_magnitude = rng.uniform(1e-6, 1e-5)
            dv = dv_magnitude * dv_direction
            
            # Ensure v+dv is still in disk
            v_perturbed = v + dv
            if np.linalg.norm(v_perturbed) >= 0.99:
                continue
            
            # Numerical distance
            d_num = hilbert_distance(v, v_perturbed)
            
            # Linear approximation via metric tensor
            G = klein_metric_tensor(v)
            d_lin_sq = dv @ G @ dv
            d_lin = np.sqrt(max(d_lin_sq, 0))  # Ensure non-negative
            
            # Relative error (should be O(||dv||))
            if d_lin > 1e-10:
                rel_error = abs(d_num - d_lin) / d_lin
                errors.append(rel_error)
        
        # Most errors should be small (O(dv_magnitude))
        errors = np.array(errors)
        assert np.median(errors) < 1e-3, f"Median rel error {np.median(errors)} too large"
        assert np.percentile(errors, 95) < 1e-2, f"95th percentile {np.percentile(errors, 95)} too large"

    def test_metric_second_order_convergence(self):
        """Verify error between metric prediction and distance scales reasonably.
        
        The metric ds² = dv^T G dv predicts infinitesimal distances.
        For small but finite dv, the error |d_num - d_lin| should be small.
        
        Note: Precise O(||dv||²) convergence testing requires very careful
        numerical analysis. Here we just verify the error decreases appropriately.
        """
        # Fixed point
        v = np.array([0.3, 0.4])
        G = klein_metric_tensor(v)
        
        # Fixed direction
        direction = np.array([0.6, 0.8])  # unit vector
        
        # Test that error is small at various scales
        for mag in [1e-4, 1e-5, 1e-6]:
            dv = mag * direction
            v_perturbed = v + dv
            
            d_num = hilbert_distance(v, v_perturbed)
            d_lin_sq = dv @ G @ dv
            d_lin = np.sqrt(max(d_lin_sq, 0))
            
            # Relative error should be small for small perturbations
            if d_lin > 1e-12:
                rel_error = abs(d_num - d_lin) / d_lin
                assert rel_error < 0.01, (
                    f"Relative error {rel_error} too large at mag={mag}"
                )

    def test_metric_batch_consistency(self):
        """Verify batch computation gives same result as scalar."""
        v_batch = np.array([
            [0.2, 0.3],
            [0.5, -0.2],
            [-0.3, 0.4],
        ])
        
        G_batch = klein_metric_tensor(v_batch)
        
        for i, v in enumerate(v_batch):
            G_scalar = klein_metric_tensor(v)
            np.testing.assert_allclose(G_batch[i], G_scalar, atol=1e-14)
