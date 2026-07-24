"""Pytest configuration for chromabloch tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Provide a seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def default_theta():
    """Provide default Theta parameters."""
    from chromabloch.params import Theta
    return Theta.default()
