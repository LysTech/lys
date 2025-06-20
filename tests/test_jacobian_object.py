import numpy as np
import pytest

from lys.objects.jacobian import Jacobian


def test_jacobian_initialization():
    """Tests that the Jacobian can be initialized correctly."""
    data = np.random.rand(10, 10, 10)
    j = Jacobian(data)
    np.testing.assert_array_equal(j.data, data)


def test_jacobian_initialization_fails_with_wrong_dims():
    """Tests that Jacobian initialization fails if data is not 3D."""
    with pytest.raises(ValueError, match="Jacobian data must be a 3D numpy array."):
        Jacobian(np.random.rand(10, 10))


def test_sample_at_vertices():
    """Tests the sample_at_vertices method of the Jacobian class."""
    # Create a simple 3x3x3 jacobian where the value is the sum of indices
    x, y, z = np.mgrid[0:3, 0:3, 0:3]
    data = x + y + z
    j = Jacobian(data)

    # Test sampling at integer coordinates (on grid points)
    vertices_on_grid = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    expected_on_grid = np.array([0, 3, 6])
    sampled_on_grid = j.sample_at_vertices(vertices_on_grid)
    np.testing.assert_allclose(sampled_on_grid, expected_on_grid)

    # Test sampling between grid points (interpolation)
    vertices_between_grid = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
    expected_between_grid = np.array([1.5, 4.5])
    sampled_between_grid = j.sample_at_vertices(vertices_between_grid)
    np.testing.assert_allclose(sampled_between_grid, expected_between_grid)

    # Test sampling outside the bounds (should return fill_value, which is 0)
    vertices_outside = np.array([[-1, 0, 0], [10, 10, 10]])
    expected_outside = np.array([0, 0])
    sampled_outside = j.sample_at_vertices(vertices_outside)
    np.testing.assert_allclose(sampled_outside, expected_outside) 