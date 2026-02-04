"""Independent validation tests.

These tests verify the implementation using independent computational methods,
providing stronger evidence of correctness than round-trip tests alone.
"""

import numpy as np
import pytest

from chromabloch.params import Theta, d65_whitepoint_lms_hpe
from chromabloch.geometry import (
    hilbert_distance,
    hilbert_distance_crossratio,
    hilbert_distance_from_origin,
)
from chromabloch.mapping import phi_theta


class TestHilbertDistanceCrossRatioValidation:
    """Validate Klein formula against cross-ratio definition.
    
    This is an INDEPENDENT validation because the two methods
    compute the same quantity via completely different formulas:
    
    1. Klein formula: arcosh((1 - ⟨p,q⟩) / sqrt((1-||p||²)(1-||q||²)))
    2. Cross-ratio: (1/2) * |log((|a⁺-p||a⁻-q|) / (|a⁺-q||a⁻-p|))|
    
    Agreement between them provides strong evidence of correctness.
    """

    def test_klein_vs_crossratio_random_pairs(self):
        """Compare Klein and cross-ratio formulas on random point pairs."""
        rng = np.random.default_rng(42)
        
        # Generate random points strictly inside disk
        for _ in range(100):
            r1 = rng.uniform(0.1, 0.8)
            r2 = rng.uniform(0.1, 0.8)
            theta1 = rng.uniform(-np.pi, np.pi)
            theta2 = rng.uniform(-np.pi, np.pi)
            
            p = np.array([r1 * np.cos(theta1), r1 * np.sin(theta1)])
            q = np.array([r2 * np.cos(theta2), r2 * np.sin(theta2)])
            
            d_klein = hilbert_distance(p, q)
            d_crossratio = hilbert_distance_crossratio(p, q)
            
            np.testing.assert_allclose(
                d_klein, d_crossratio, rtol=1e-8,
                err_msg=f"Mismatch at p={p}, q={q}"
            )

    def test_klein_vs_crossratio_near_origin(self):
        """Compare formulas for points near origin."""
        rng = np.random.default_rng(123)
        
        for _ in range(50):
            # One point near origin
            p = rng.uniform(-0.1, 0.1, size=2)
            # Other point anywhere
            r = rng.uniform(0.1, 0.9)
            theta = rng.uniform(-np.pi, np.pi)
            q = np.array([r * np.cos(theta), r * np.sin(theta)])
            
            d_klein = hilbert_distance(p, q)
            d_crossratio = hilbert_distance_crossratio(p, q)
            
            np.testing.assert_allclose(d_klein, d_crossratio, rtol=1e-7)

    def test_klein_vs_crossratio_near_boundary(self):
        """Compare formulas for points near boundary."""
        rng = np.random.default_rng(456)
        
        for _ in range(50):
            # Both points near boundary
            r1 = rng.uniform(0.85, 0.95)
            r2 = rng.uniform(0.85, 0.95)
            theta1 = rng.uniform(-np.pi, np.pi)
            theta2 = rng.uniform(-np.pi, np.pi)
            
            p = np.array([r1 * np.cos(theta1), r1 * np.sin(theta1)])
            q = np.array([r2 * np.cos(theta2), r2 * np.sin(theta2)])
            
            d_klein = hilbert_distance(p, q)
            d_crossratio = hilbert_distance_crossratio(p, q)
            
            # Slightly looser tolerance near boundary
            np.testing.assert_allclose(d_klein, d_crossratio, rtol=1e-6)

    def test_klein_vs_crossratio_collinear_with_origin(self):
        """Compare formulas for points collinear with origin."""
        # These should match distance from origin formula
        for r1 in [0.2, 0.5, 0.8]:
            for r2 in [0.3, 0.6, 0.9]:
                p = np.array([r1, 0.0])
                q = np.array([r2, 0.0])
                
                d_klein = hilbert_distance(p, q)
                d_crossratio = hilbert_distance_crossratio(p, q)
                
                # Also compare with direct computation
                d_direct = np.abs(hilbert_distance_from_origin(p) - 
                                 hilbert_distance_from_origin(q))
                
                np.testing.assert_allclose(d_klein, d_crossratio, rtol=1e-10)
                np.testing.assert_allclose(d_klein, d_direct, rtol=1e-10)


class TestTrueZeroBehavior:
    """Test behavior with exact zero LMS values.
    
    When ε > 0, the mapping should handle exact zeros gracefully.
    Black pixels should map to the origin when whitepoint-calibrated.
    """

    def test_black_pixel_with_epsilon(self):
        """Black (0,0,0) should map to origin when ε > 0."""
        # Use D65-calibrated parameters
        L_w, M_w, S_w = d65_whitepoint_lms_hpe()
        theta = Theta.from_whitepoint(L_w, M_w, S_w, epsilon=0.01)
        
        # True black
        lms_black = np.array([0.0, 0.0, 0.0])
        v = phi_theta(lms_black, theta)
        
        # Should map to origin
        np.testing.assert_allclose(v, [0.0, 0.0], atol=1e-10)

    def test_no_nan_inf_with_zeros(self):
        """Ensure no NaN or Inf when processing zeros."""
        theta = Theta(epsilon=0.01)  # Must have ε > 0
        
        # Various zero-containing inputs
        test_cases = [
            np.array([0.0, 0.0, 0.0]),  # Pure black
            np.array([1.0, 0.0, 0.0]),  # Only L
            np.array([0.0, 1.0, 0.0]),  # Only M
            np.array([0.0, 0.0, 1.0]),  # Only S
            np.array([1.0, 1.0, 0.0]),  # No S
            np.array([0.0, 0.0, 1.0]),  # Only S
        ]
        
        for lms in test_cases:
            v = phi_theta(lms, theta)
            assert not np.any(np.isnan(v)), f"NaN for LMS={lms}"
            assert not np.any(np.isinf(v)), f"Inf for LMS={lms}"
            assert np.linalg.norm(v) < 1.0, f"Outside disk for LMS={lms}"

    def test_epsilon_zero_requires_strict_positive(self):
        """With ε=0, zero inputs cause division by zero."""
        theta = Theta(epsilon=0.0)
        
        # This SHOULD produce inf or nan (or be caught)
        lms_black = np.array([0.0, 0.0, 0.0])
        
        # The mapping will produce nan/inf when Y=0 and ε=0
        v = phi_theta(lms_black, theta)
        
        # We expect nan or inf (division by zero)
        assert np.any(np.isnan(v)) or np.any(np.isinf(v)), (
            "Expected nan/inf for black with ε=0"
        )

    def test_near_zero_lms_stability(self):
        """Test stability with very small (but positive) LMS."""
        theta = Theta(epsilon=0.01)
        
        # Very small values
        tiny_values = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
        
        for tiny in tiny_values:
            lms = np.array([tiny, tiny, tiny])
            v = phi_theta(lms, theta)
            
            assert not np.any(np.isnan(v)), f"NaN for tiny={tiny}"
            assert not np.any(np.isinf(v)), f"Inf for tiny={tiny}"
            # Gray should be near origin
            assert np.linalg.norm(v) < 0.1, f"Gray too saturated for tiny={tiny}"

    def test_batch_with_zeros(self):
        """Test batch processing with some zeros mixed in."""
        theta = Theta(epsilon=0.01)
        
        lms_batch = np.array([
            [0.0, 0.0, 0.0],   # Black
            [1.0, 1.0, 1.0],   # Gray
            [1.0, 0.0, 0.0],   # Only L
            [0.0, 1.0, 1.0],   # M+S, no L
            [0.5, 0.5, 0.5],   # Medium gray
        ])
        
        v_batch = phi_theta(lms_batch, theta)
        
        assert v_batch.shape == (5, 2)
        assert not np.any(np.isnan(v_batch))
        assert not np.any(np.isinf(v_batch))
        assert np.all(np.linalg.norm(v_batch, axis=-1) < 1.0)


class TestGamutFeasibility:
    """Test reconstruction feasibility across the disk."""

    def test_random_v_reconstruction_feasibility(self):
        """Check what fraction of random v points are reconstructible."""
        from chromabloch.reconstruction import positivity_conditions, reconstruct_lms
        
        theta = Theta.default()
        rng = np.random.default_rng(789)
        
        n_samples = 1000
        n_feasible = 0
        
        for _ in range(n_samples):
            # Random point in disk
            r = rng.uniform(0, 0.99)
            angle = rng.uniform(-np.pi, np.pi)
            v = np.array([r * np.cos(angle), r * np.sin(angle)])
            
            # Random luminance
            Y = rng.uniform(0.1, 2.0)
            
            # Check feasibility (returns a dict)
            cond = positivity_conditions(v, Y, theta)
            L_ok, M_ok, S_ok = cond["L_pos"], cond["M_pos"], cond["S_pos"]
            
            if L_ok and M_ok and S_ok:
                n_feasible += 1
                # Verify reconstruction produces positive LMS
                lms = reconstruct_lms(v, Y, theta)
                assert np.all(lms > -1e-10), f"Negative LMS despite passing conditions"
        
        feasibility_rate = n_feasible / n_samples
        print(f"Feasibility rate: {feasibility_rate*100:.1f}%")
        
        # Expect at least 30% to be feasible (depends on Y range)
        assert feasibility_rate > 0.2, f"Feasibility too low: {feasibility_rate}"

    def test_attainable_region_points_always_feasible(self):
        """Points from actual LMS should always be reconstructible."""
        from chromabloch.reconstruction import positivity_conditions, reconstruct_lms
        
        theta = Theta.default()
        rng = np.random.default_rng(101)
        
        # Generate random LMS and map forward
        for _ in range(100):
            lms = 10.0 ** rng.uniform(-0.5, 0.5, size=3)
            v = phi_theta(lms, theta)
            Y = theta.w_L * lms[0] + theta.w_M * lms[1]
            
            # These MUST be feasible (returns a dict)
            cond = positivity_conditions(v, Y, theta)
            L_ok, M_ok, S_ok = cond["L_pos"], cond["M_pos"], cond["S_pos"]
            assert L_ok and M_ok and S_ok, (
                f"Forward-mapped point not feasible: LMS={lms}, v={v}, Y={Y}"
            )
            
            # Reconstruction should match
            lms_rec = reconstruct_lms(v, Y, theta)
            np.testing.assert_allclose(lms_rec, lms, rtol=1e-8)
