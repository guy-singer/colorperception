"""Tests for opponent transform module."""

import numpy as np
import pytest

from chromabloch.params import Theta
from chromabloch.opponent import opponent_transform, A_theta, det_A_theta


class TestOpponentTransform:
    """Tests for opponent_transform function."""

    def test_basic_computation(self):
        """Test basic opponent transform computation."""
        theta = Theta.default()
        lms = np.array([1.0, 1.0, 1.0])

        Y, O1, O2 = opponent_transform(lms, theta)

        # Y = w_L*L + w_M*M = 1*1 + 1*1 = 2
        assert np.isclose(Y, 2.0)
        # O1 = L - gamma*M = 1 - 1*1 = 0
        assert np.isclose(O1, 0.0)
        # O2 = S - beta*(L+M) = 1 - 0.5*2 = 0
        assert np.isclose(O2, 0.0)

    def test_achromatic_point(self):
        """Test that achromatic LMS gives O1=O2=0."""
        theta = Theta.default()
        # Achromatic: L = gamma*M, S = beta*(1+gamma)*M
        M = 1.0
        L = theta.gamma * M
        S = theta.beta * (1 + theta.gamma) * M
        lms = np.array([L, M, S])

        Y, O1, O2 = opponent_transform(lms, theta)

        assert Y > 0
        assert np.isclose(O1, 0.0, atol=1e-10)
        assert np.isclose(O2, 0.0, atol=1e-10)

    def test_batch_processing(self):
        """Test batch processing of multiple LMS values."""
        theta = Theta.default()
        lms = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 0.5], [0.5, 2.0, 1.5]])

        Y, O1, O2 = opponent_transform(lms, theta)

        assert Y.shape == (3,)
        assert O1.shape == (3,)
        assert O2.shape == (3,)

    def test_input_validation(self):
        """Test that invalid input shapes raise errors."""
        theta = Theta.default()
        lms_bad = np.array([1.0, 2.0])  # Wrong shape

        with pytest.raises(ValueError):
            opponent_transform(lms_bad, theta)


class TestATheta:
    """Tests for A_theta matrix function."""

    def test_matrix_shape(self):
        """Test that A_theta returns 3x3 matrix."""
        theta = Theta.default()
        A = A_theta(theta)

        assert A.shape == (3, 3)

    def test_matrix_entries(self):
        """Test matrix entries match definition."""
        theta = Theta(w_L=1.5, w_M=0.8, gamma=1.2, beta=0.6, epsilon=0.01, kappa=1.0)
        A = A_theta(theta)

        expected = np.array([
            [1.5, 0.8, 0.0],
            [1.0, -1.2, 0.0],
            [-0.6, -0.6, 1.0],
        ])
        np.testing.assert_allclose(A, expected)

    def test_matrix_times_lms_equals_transform(self):
        """Test that A @ [L,M,S] equals opponent_transform output."""
        theta = Theta.default()
        lms = np.array([0.8, 1.2, 0.5])

        A = A_theta(theta)
        result_matrix = A @ lms

        Y, O1, O2 = opponent_transform(lms, theta)
        result_func = np.array([Y, O1, O2])

        np.testing.assert_allclose(result_matrix, result_func)


class TestDetATheta:
    """Tests for determinant function."""

    def test_determinant_formula(self):
        """Test det(A_theta) = -(w_L*gamma + w_M)."""
        theta = Theta(w_L=1.5, w_M=0.8, gamma=1.2, beta=0.6, epsilon=0.01, kappa=1.0)

        det = det_A_theta(theta)
        expected = -(1.5 * 1.2 + 0.8)

        assert np.isclose(det, expected)

    def test_determinant_equals_minus_delta(self):
        """Test det = -Delta."""
        theta = Theta.default()

        det = det_A_theta(theta)

        assert np.isclose(det, -theta.Delta)

    def test_determinant_nonzero(self):
        """Test that determinant is always nonzero for valid params."""
        theta = Theta.default()

        det = det_A_theta(theta)

        assert det != 0
